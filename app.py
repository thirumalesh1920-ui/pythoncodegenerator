
import streamlit as st
import torch
import torch.nn as nn
import json
import re

MODEL2_PATH = "model2_attention_seq2seq.pt"
Q_STOI_PATH = "q_stoi.json"
Q_ITOS_PATH = "q_itos.json"
CODE_STOI_PATH = "code_stoi.json"
CODE_ITOS_PATH = "code_itos.json"

PAD = "<pad>"
SOS = "<sos>"
EOS = "<eos>"
UNK = "<unk>"

MAX_QUESTION_LEN = 100
MAX_CODE_LEN = 300
DEVICE = torch.device("cpu")


# ---------------------------
# Model 2 classes
# ---------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class EncoderAttn(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(hidden_dim + emb_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)

        context = torch.bmm(attn_weights, encoder_outputs)
        rnn_input = torch.cat((embedded, context), dim=2)

        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context), dim=1))
        return prediction, hidden, cell


class Seq2SeqAttention(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output
            top1 = output.argmax(1)

            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else top1

        return outputs


# ---------------------------
# Helper functions
# ---------------------------
def tokenize_question(text):
    return text.lower().strip().split()


def token_from_itos(itos_obj, idx):
    if isinstance(itos_obj, list):
        if 0 <= idx < len(itos_obj):
            return itos_obj[idx]
        return UNK

    if isinstance(itos_obj, dict):
        return itos_obj.get(str(idx), itos_obj.get(idx, UNK))

    return UNK


def decode_ids(ids, code_itos):
    tokens = []
    for idx in ids:
        tok = token_from_itos(code_itos, idx)
        if tok in [PAD, SOS, EOS]:
            continue
        tokens.append(tok)

    return " ".join(tokens)


@st.cache_resource
def load_model2_and_vocab():
    with open(Q_STOI_PATH, "r", encoding="utf-8") as f:
        q_stoi = json.load(f)

    with open(CODE_STOI_PATH, "r", encoding="utf-8") as f:
        code_stoi = json.load(f)

    with open(CODE_ITOS_PATH, "r", encoding="utf-8") as f:
        code_itos = json.load(f)

    state_dict = torch.load(MODEL2_PATH, map_location=DEVICE)

    # infer dimensions from state dict
    input_dim = state_dict["encoder.embedding.weight"].shape[0]
    enc_emb_dim = state_dict["encoder.embedding.weight"].shape[1]
    hidden_dim = state_dict["encoder.rnn.weight_hh_l0"].shape[1]
    output_dim = state_dict["decoder.embedding.weight"].shape[0]
    dec_emb_dim = state_dict["decoder.embedding.weight"].shape[1]

    attention = Attention(hidden_dim)
    encoder2 = EncoderAttn(input_dim, enc_emb_dim, hidden_dim)
    decoder2 = DecoderWithAttention(output_dim, dec_emb_dim, hidden_dim, attention)

    model2 = Seq2SeqAttention(encoder2, decoder2, DEVICE).to(DEVICE)
    model2.load_state_dict(state_dict)
    model2.eval()

    return model2, q_stoi, code_stoi, code_itos


def generate_code_model2(model, question_text, q_stoi, code_stoi, code_itos, max_len=MAX_CODE_LEN):
    model.eval()

    tokens = tokenize_question(question_text)
    unk_id = q_stoi.get(UNK, 3)
    sos_id_q = q_stoi.get(SOS, 1)
    eos_id_q = q_stoi.get(EOS, 2)

    ids = [q_stoi.get(tok, unk_id) for tok in tokens]
    ids = [sos_id_q] + ids[:MAX_QUESTION_LEN - 2] + [eos_id_q]

    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    generated_ids = []

    sos_id_code = code_stoi.get(SOS, 1)
    eos_id_code = code_stoi.get(EOS, 2)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        input_token = torch.tensor([sos_id_code], dtype=torch.long).to(DEVICE)

        for _ in range(max_len):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            pred_token = output.argmax(1).item()

            if pred_token == eos_id_code:
                break

            generated_ids.append(pred_token)
            input_token = torch.tensor([pred_token], dtype=torch.long).to(DEVICE)

    return decode_ids(generated_ids, code_itos)


# ---------------------------
# Load resources
# ---------------------------
model2, q_stoi, code_stoi, code_itos = load_model2_and_vocab()


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="ChatTPCG", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

st.sidebar.markdown("## 🤖 Models")

model_choice = st.sidebar.radio(
    "Select Model",
    ["Model 1", "Model 2", "Model 3"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🕒 Recent Chats")

if len(st.session_state.history) == 0:
    st.sidebar.write("No recent chats yet.")
else:
    for i, chat in enumerate(st.session_state.history[-5:], start=1):
        st.sidebar.write(f"{i}. {chat}")

st.sidebar.markdown("---")

if st.sidebar.button("🗑 Clear History"):
    st.session_state.history = []
    st.session_state.generated_code = ""
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.write("👤 Profile")
st.sidebar.write("❓ Help Center")
st.sidebar.write("⚙️ Settings")

st.markdown(
    "<h1 style='text-align: center;'>💬 ChatTPCG</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Welcome to Python Code Generator!</p>",
    unsafe_allow_html=True
)

st.write("")
st.markdown("### Ask your coding task")

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Enter your Python task:",
        placeholder="Type something like: give me prime numbers code",
        label_visibility="collapsed"
    )

with col2:
    generate = st.button("Send ➤")

if generate:
    if user_input.strip():
        with st.spinner("Generating code..."):
            try:
                if model_choice == "Model 2":
                    output = generate_code_model2(
                        model2,
                        user_input,
                        q_stoi,
                        code_stoi,
                        code_itos
                    )
                    if not output.strip():
                        output = "Model 2 returned empty output."
                else:
                    output = f"{model_choice} is not connected yet. Currently only Model 2 is connected."
            except Exception as e:
                output = f"Error running {model_choice}: {str(e)}"

        st.session_state.history.append(user_input)
        st.session_state.generated_code = output
    else:
        st.warning("Please enter something.")

if st.session_state.generated_code:
    st.markdown("### 💡 Generated Output")

    st.markdown("**You:**")
    if len(st.session_state.history) > 0:
        st.info(st.session_state.history[-1])

    st.markdown("**ChatTPCG:**")
    st.code(st.session_state.generated_code, language="python")
