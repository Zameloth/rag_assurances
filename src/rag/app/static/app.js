// Hand-rolled chat scaffolding (SPEC §13.6) — everything else is HTMX attributes.
document.addEventListener("DOMContentLoaded", () => {
  const form = document.querySelector("form.ask");
  const question = form.querySelector("textarea");

  // Enter sends, Shift+Enter breaks the line.
  question.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  // Clear the question only once an answer came back, so a failed send keeps the text.
  form.addEventListener("htmx:afterRequest", (event) => {
    if (event.detail.successful) {
      form.reset();
      question.focus();
    }
  });
});
