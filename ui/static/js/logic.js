document.addEventListener("DOMContentLoaded", function() {
    document.getElementById("send-btn").addEventListener("click", sendMessage);
});

function sendMessage() {
    let input = document.getElementById("user-input");
    let message = input.value.trim();
    if (!message) return;

    let chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<p><strong>You:</strong> ${message}</p>`;
    input.value = "";

    fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        chatBox.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        chatBox.scrollTop = chatBox.scrollHeight;
    });
}
