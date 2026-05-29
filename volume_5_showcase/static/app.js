const promptInput = document.getElementById('promptInput');
const generateBtn = document.getElementById('generateBtn');
const stopBtn = document.getElementById('stopBtn');
const chatBox = document.getElementById('chatBox');
const probabilityBars = document.getElementById('probabilityBars');

let isGenerating = false;
let currentPrompt = "";
let currentMessageElement = null;

generateBtn.addEventListener('click', startGeneration);
promptInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') startGeneration();
});
stopBtn.addEventListener('click', () => {
    isGenerating = false;
});

function startGeneration() {
    const text = promptInput.value.trim();
    if (!text || isGenerating) return;

    // Add User Message
    addMessage(text, 'user');
    promptInput.value = '';

    // Setup AI Message
    currentMessageElement = addMessage(text, 'ai');
    currentPrompt = text;
    isGenerating = true;
    
    // Toggle Buttons
    generateBtn.style.display = 'none';
    stopBtn.style.display = 'block';
    
    // Start the Autoregressive Loop
    generateNextToken();
}

function addMessage(text, sender) {
    const msg = document.createElement('div');
    msg.className = `message ${sender}`;
    msg.textContent = text;
    chatBox.appendChild(msg);
    chatBox.scrollTop = chatBox.scrollHeight;
    return msg;
}

async function generateNextToken() {
    if (!isGenerating) {
        finishGeneration();
        return;
    }

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt: currentPrompt })
        });

        if (!response.ok) throw new Error("API Error");

        const data = await response.json();
        const nextChar = data.next_char;
        const top5 = data.top_5;

        // Update Text
        currentPrompt += nextChar;
        currentMessageElement.textContent = currentPrompt;
        chatBox.scrollTop = chatBox.scrollHeight;

        // Update Visualizer
        renderProbabilities(top5);

        // We limit to max 500 characters so it doesn't run forever
        // Or if the user hits stop
        if (currentPrompt.length > 500) {
            isGenerating = false;
        }

        // Loop using requestAnimationFrame for smooth UI updates
        requestAnimationFrame(generateNextToken);

    } catch (error) {
        console.error("Generation Error:", error);
        finishGeneration();
    }
}

function renderProbabilities(top5) {
    probabilityBars.innerHTML = ''; // Clear existing

    // We want the highest probability to have the most vivid color
    top5.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = 'prob-item';
        
        div.innerHTML = `
            <div class="prob-char">${item.char}</div>
            <div class="prob-bar-container">
                <!-- Set width dynamically based on prob percentage -->
                <div class="prob-bar" style="width: ${item.prob}%; opacity: ${1 - (index * 0.15)}"></div>
            </div>
            <div class="prob-value">${item.prob.toFixed(1)}%</div>
        `;
        
        probabilityBars.appendChild(div);
    });
}

function finishGeneration() {
    isGenerating = false;
    generateBtn.style.display = 'block';
    stopBtn.style.display = 'none';
    // Don't clear the probabilities, let them see the final thought!
}
