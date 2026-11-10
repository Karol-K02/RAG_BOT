"""
DocChat - a desktop chat app for a local model (LM Studio, Ollama...).
Optionally answers from your own documents (RAG), using mini_rag.py.

Install:
    pip install -r requirements.txt

Run:
    python app.py
    (without a console window: pythonw app.py)

Settings: environment variables LLM_BASE_URL, LLM_MODEL, LLM_API_KEY,
or change the defaults below.
"""
import os
import queue
import sys
import threading

import customtkinter as ctk
from openai import OpenAI

# --- settings -------------------------------------------------------------
BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LLM_API_KEY", "lm-studio")
MODEL = os.getenv("LLM_MODEL", "qwen/qwen3.5-9b")
CHAT_PROMPT = ("You are DocChat, a helpful assistant. Answer concisely and to the point, "
               "in the language the user writes in.")
MAX_HISTORY = 20  # how many recent messages to send to the model

# --- colours --------------------------------------------------------------
BG, PANEL, ACCENT = "#0b1020", "#131a2e", "#22d3ee"
USER_COLOR, BOT_COLOR, MUTED, ERR = "#7dd3fc", "#c4b5fd", "#94a3b8", "#f87171"

# Work from the script's folder, so mini_rag.py and the docs/ folder are found
APP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)


class ThinkFilter:
    """Removes <think>...</think> blocks (Qwen3 etc.) from the token stream."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> str:
        self.buf += text
        out = ""
        while True:
            tag = self.CLOSE if self.inside else self.OPEN
            i = self.buf.find(tag)
            if i != -1:
                if not self.inside:
                    out += self.buf[:i]
                self.buf = self.buf[i + len(tag):]
                self.inside = not self.inside
                continue
            keep = 0  # the end of the buffer that might be the start of a tag
            for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                if tag.startswith(self.buf[-n:]):
                    keep = n
                    break
            emit = self.buf[: len(self.buf) - keep]
            if not self.inside:
                out += emit
            self.buf = self.buf[len(self.buf) - keep:]
            return out

    def flush(self) -> str:
        out = "" if self.inside else self.buf
        self.buf = ""
        return out


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("DocChat")
        self.geometry("920x720")
        self.minsize(640, 480)
        self.configure(fg_color=BG)

        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
        self.history: list[dict] = []
        self.busy = False
        self.rag_on = False
        self.index = None
        self.rag_prompt = ""
        self.label = None
        self.events: queue.Queue = queue.Queue()

        self._build_ui()
        self._intro()
        self.after(50, self._poll)
        threading.Thread(target=self._check_connection, daemon=True).start()

    # --- interface ---------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(16, 8))
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkFrame(header, fg_color="transparent")
        title.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(title, text="DocChat", text_color=ACCENT,
                     font=ctk.CTkFont(size=30, weight="bold")).pack(anchor="w")
        ctk.CTkLabel(title, text=f"model: {MODEL}", text_color=MUTED,
                     font=ctk.CTkFont(size=12)).pack(anchor="w")

        right = ctk.CTkFrame(header, fg_color="transparent")
        right.grid(row=0, column=2, sticky="e")
        self.status = ctk.CTkLabel(right, text="● connecting...", text_color=MUTED)
        self.status.pack(anchor="e")
        self.rag_switch = ctk.CTkSwitch(right, text="Documents (RAG)",
                                        command=self._toggle_rag, progress_color=ACCENT)
        self.rag_switch.pack(anchor="e", pady=(6, 0))

        self.chat = ctk.CTkTextbox(self, wrap="word", fg_color=PANEL, corner_radius=14,
                                   font=ctk.CTkFont(size=15), text_color="#e5e7eb")
        self.chat.grid(row=1, column=0, sticky="nsew", padx=20, pady=8)
        self.chat.tag_config("user_name", foreground=USER_COLOR)
        self.chat.tag_config("bot_name", foreground=BOT_COLOR)
        self.chat.tag_config("muted", foreground=MUTED)
        self.chat.tag_config("err", foreground=ERR)
        self.chat.configure(state="disabled")

        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=20, pady=(8, 16))
        bottom.grid_columnconfigure(0, weight=1)
        self.entry = ctk.CTkEntry(bottom, height=46, corner_radius=12, fg_color=PANEL,
                                  border_color=ACCENT, font=ctk.CTkFont(size=15),
                                  placeholder_text="Type a message and press Enter...")
        self.entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.entry.bind("<Return>", self._send)
        self.send_btn = ctk.CTkButton(bottom, text="Send", width=100, height=46,
                                      corner_radius=12, fg_color=ACCENT, text_color="#062a33",
                                      hover_color="#67e8f9", command=self._send)
        self.send_btn.grid(row=0, column=1, padx=(0, 10))
        ctk.CTkButton(bottom, text="Clear", width=90, height=46, corner_radius=12,
                      fg_color="#1e293b", hover_color="#334155",
                      command=self._reset).grid(row=0, column=2)
        self.entry.focus()

    def _append(self, text: str, tag: str | None = None):
        self.chat.configure(state="normal")
        self.chat.insert("end", text, tag) if tag else self.chat.insert("end", text)
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _intro(self):
        self._append("Hi! I'm DocChat. Ask me anything, or switch on \"Documents (RAG)\" "
                     "to get answers from the files in the docs folder.\n\n",
                     "muted")

    def _set_status(self, text: str, color: str):
        self.status.configure(text=text, text_color=color)

    # --- actions -----------------------------------------------------------
    def _reset(self):
        if self.busy:
            return
        self.history.clear()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._intro()

    def _send(self, event=None):
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        self.entry.delete(0, "end")
        self._append("You\n", "user_name")
        self._append(text + "\n\n")
        self._append("DocChat\n", "bot_name")
        self.history.append({"role": "user", "content": text})
        self.busy = True
        self.send_btn.configure(state="disabled")
        threading.Thread(target=self._worker, args=(text,), daemon=True).start()

    def _toggle_rag(self):
        if not self.rag_switch.get():
            self.rag_on = False
            return
        if self.index is not None:
            self.rag_on = True
            return
        self.rag_switch.configure(state="disabled")
        self._set_status("● indexing documents...", ACCENT)
        threading.Thread(target=self._build_index, daemon=True).start()

    # --- worker threads (they don't touch the interface, they only send events) ---
    def _check_connection(self):
        try:
            self.client.models.list()
            self.events.put(("status", "● connected to the server", "#4ade80"))
        except Exception:
            self.events.put(("status", "● no connection to the server", ERR))

    def _build_index(self):
        try:
            from mini_rag import SYSTEM_PROMPT, Index, label, load_documents
            pages = load_documents()
            if not pages:
                raise RuntimeError("the docs folder is empty")
            index = Index(pages)
            self.events.put(("rag_ready", index, SYSTEM_PROMPT, label, len(index.chunks)))
        except Exception as e:
            self.events.put(("rag_failed", str(e)))

    def _worker(self, question: str):
        try:
            system = CHAT_PROMPT
            messages = list(self.history[-MAX_HISTORY:])
            hits = []
            if self.rag_on and self.index is not None:
                hits = self.index.search(question)
                context = "\n\n".join(
                    f'<source name="{self.label(h)}">\n{h["text"]}\n</source>' for h in hits
                )
                system = self.rag_prompt
                messages[-1] = {"role": "user",
                                "content": f"Sources:\n{context}\n\nQuestion: {question}"}

            stream = self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=0.1 if hits else 0.7,
                max_tokens=3000,
                # No "thinking" phase: Qwen3.5 could spend about 2,000 tokens and a minute and a half
                # before the first word, and the thinking counted towards the max_tokens limit.
                reasoning_effort="none",
                stream=True,
            )
            think, reply, finish = ThinkFilter(), "", None
            for chunk in stream:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                piece = think.feed(choice.delta.content or "")
                if piece:
                    reply += piece
                    self.events.put(("token", piece))
                if choice.finish_reason:
                    finish = choice.finish_reason
            tail = think.flush()
            if tail:
                reply += tail
                self.events.put(("token", tail))
            self.events.put(("done", reply, finish, hits))
        except Exception as e:
            self.events.put(("error", str(e)))

    # --- handling events in the main thread --------------------------------
    def _poll(self):
        try:
            while True:
                ev = self.events.get_nowait()
                kind = ev[0]
                if kind == "token":
                    self._append(ev[1])
                elif kind == "done":
                    _, reply, finish, hits = ev
                    self._append("\n")
                    if finish == "length":
                        self._append("[answer cut off by the token limit]\n", "err")
                    if hits:
                        names = list(dict.fromkeys(self.label(h) for h in hits))
                        # show only the sources actually cited in the answer
                        cited = [n for n in names if n in reply]
                        self._append("Sources: " + "; ".join(cited or names) + "\n", "muted")
                    self._append("\n")
                    self.history.append({"role": "assistant", "content": reply})
                    self._finish_turn()
                elif kind == "error":
                    self._append(f"\nError: {ev[1]}\nCheck that the server is running and the model is loaded.\n\n", "err")
                    if self.history and self.history[-1]["role"] == "user":
                        self.history.pop()
                    self._set_status("● no connection to the server", ERR)
                    self._finish_turn()
                elif kind == "status":
                    self._set_status(ev[1], ev[2])
                elif kind == "rag_ready":
                    _, self.index, self.rag_prompt, self.label, n = ev
                    self.rag_on = True
                    self.rag_switch.configure(state="normal")
                    self._set_status(f"● RAG ready ({n} fragments)", "#4ade80")
                elif kind == "rag_failed":
                    self.rag_on = False
                    self.rag_switch.deselect()
                    self.rag_switch.configure(state="normal")
                    self._set_status("● RAG unavailable", ERR)
                    self._append(f"Couldn't switch on document mode: {ev[1]}\n"
                                 "Needed: mini_rag.py next to this file and "
                                 "pip install sentence-transformers numpy pymupdf\n\n", "err")
        except queue.Empty:
            pass
        self.after(40, self._poll)

    def _finish_turn(self):
        self.busy = False
        self.send_btn.configure(state="normal")
        self.entry.focus()


if __name__ == "__main__":
    App().mainloop()
