// Browser-local chat state only: the backend receives the current prompt and
// a session id. Nothing is persisted server-side (Plan 3 v1 scope).

const log = document.getElementById("chat-log");
const form = document.getElementById("chat-form");
const input = document.getElementById("message");
const sendButton = document.getElementById("send");
const resetButton = document.getElementById("reset");

let sessionId = sessionStorage.getItem("tinygpt-session") || newSessionId();

function newSessionId() {
  const id = crypto.randomUUID();
  sessionStorage.setItem("tinygpt-session", id);
  return id;
}

function addBubble(text, className) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${className}`;
  bubble.textContent = text;
  log.appendChild(bubble);
  log.scrollTop = log.scrollHeight;
  return bubble;
}

function addMeta(text) {
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = text;
  log.appendChild(meta);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) return;

  addBubble(message, "user");
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  const pending = addBubble("writing a story…", "model pending");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    const body = await response.json();
    if (!response.ok) {
      const detail = body?.error?.message || `request failed (${response.status})`;
      pending.remove();
      addBubble(detail, "error");
      return;
    }
    pending.remove();
    addBubble(body.reply, "model");
    addMeta(
      `${body.model_version} · ${body.output_token_count} tokens · ` +
        `${Math.round(body.latency_ms)} ms`
    );
  } catch (err) {
    pending.remove();
    addBubble("network error; please try again", "error");
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
});

resetButton.addEventListener("click", () => {
  log.replaceChildren();
  sessionId = newSessionId();
  input.focus();
});

input.focus();
