// Hand-rolled chat scaffolding (SPEC §13.6). A question is POSTed to /ask/stream, whose SSE
// body reports each pipeline stage as it runs (SPEC §13.2) and ends with the rendered
// exchange, which HTMX swaps in. `fetch` rather than `EventSource`: the latter can only GET,
// and the question and history belong in a request body, not a URL.
const UNAVAILABLE = "Le service est momentanément indisponible. Réessayez dans un instant.";

document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form.ask");
  const question = form.querySelector("textarea");
  const button = form.querySelector("button");
  const conversation = document.querySelector("#conversation");
  const reset = document.querySelector("button.reset");

  // Enter sends, Shift+Enter breaks the line.
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  // The history is the rendered exchanges' hidden inputs, so this is the whole reset.
  reset.addEventListener("click", () => {
    conversation.replaceChildren();
    question.focus();
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (form.classList.contains("is-busy")) return;

    // Stateless (SPEC §13.4): the history is whatever the rendered exchanges carry.
    const body = new URLSearchParams(new FormData(form));
    for (const input of conversation.querySelectorAll("input[name^='history_']")) {
      body.append(input.name, input.value);
    }

    const pending = pendingExchange(question.value);
    conversation.append(pending);
    scrollToLatest();
    form.classList.add("is-busy");
    // Clearing mid-answer would detach the exchange the answer is about to replace.
    button.disabled = reset.disabled = true;

    try {
      const { event: outcome, html } = await ask(body, pending.querySelector(".stages"));
      htmx.swap(pending, html, { swapStyle: "outerHTML" });
      // Clear the question only once an answer came back, so a failed send keeps the text.
      if (outcome === "result") form.reset();
    } catch (error) {
      pending.replaceWith(errorExchange(question.value, error.message || UNAVAILABLE));
    } finally {
      form.classList.remove("is-busy");
      button.disabled = reset.disabled = false;
      question.focus();
      scrollToLatest();
    }
  });
});

// The page's own bottom, not the exchange's: the sticky form would cover the latter.
function scrollToLatest() {
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
}

// Reads the stream until the `result` or `error` event, marking each `stage` as it arrives.
async function ask(body, stages) {
  const response = await fetch("/ask/stream", {
    method: "POST",
    body,
    headers: { Accept: "text/event-stream" },
  });
  // A sleeping container is answered by Sablier's waiting page — HTML, not a stream.
  if (!response.headers.get("content-type")?.startsWith("text/event-stream")) {
    throw new Error(response.ok ? "Le service démarre. Réessayez dans un instant." : UNAVAILABLE);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) throw new Error(UNAVAILABLE);
    buffer += value.replace(/\r\n?/g, "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const { event, data } = parseEvent(buffer.slice(0, end));
      buffer = buffer.slice(end + 2);
      if (event === "stage") showStage(stages, JSON.parse(data));
      else if (event === "result" || event === "error") return { event, html: data };
    }
  }
}

function parseEvent(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(line.startsWith("data: ") ? 6 : 5));
    // Anything else is a `:` comment — the server's heartbeat.
  }
  return { event, data: data.join("\n") };
}

function showStage(stages, { stage, label }) {
  for (const done of stages.querySelectorAll(".active")) done.className = "done";
  const item = document.createElement("li");
  item.className = "active";
  item.dataset.stage = stage;
  item.textContent = label;
  stages.append(item);
}

function exchange(text, state) {
  const article = document.createElement("article");
  article.className = "exchange";
  const asked = document.createElement("p");
  asked.className = "question";
  asked.textContent = text;
  const answer = document.createElement("div");
  answer.className = "answer";
  answer.dataset.state = state;
  article.append(asked, answer);
  return article;
}

function pendingExchange(text) {
  const article = exchange(text, "en_cours");
  const stages = document.createElement("ol");
  stages.className = "stages";
  stages.setAttribute("aria-label", "Progression");
  article.querySelector(".answer").append(stages);
  return article;
}

function errorExchange(text, message) {
  const article = exchange(text, "erreur");
  article.classList.add("exchange-error");
  const error = document.createElement("p");
  error.className = "error";
  error.textContent = message;
  article.querySelector(".answer").append(error);
  return article;
}
