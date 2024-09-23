const backend = {
    sendMessage: function(blockId, message) {
        // Simulating backend message
        setTimeout(() => {
            addChatMessage(blockId, message);
        }, 2000);
    }
};

function addChatMessage(blockId, message) {
    const chatBlock = document.querySelector(`.${blockId} .chat-messages`);
    const chatMessage = document.createElement('div');
    chatMessage.textContent = message;
    chatMessage.classList.add('chat-message');
    chatBlock.appendChild(chatMessage);
}

// Function to add thumbs up and thumbs down icons to commentary messages
function addThumbsIconsToCommentary() {
    const commentaryMessages = document.querySelectorAll('.right-block .chat-message');

    commentaryMessages.forEach(message => {
        const thumbsUpIcon = document.createElement('i');
        thumbsUpIcon.className = 'far fa-thumbs-up thumbs-up-icon';
        thumbsUpIcon.addEventListener('click', () => {
            message.classList.add('liked');
            thumbsUpIcon.style.display = 'none';
            thumbsDownIcon.style.display = 'none';
        });

        const thumbsDownIcon = document.createElement('i');
        thumbsDownIcon.className = 'far fa-thumbs-down thumbs-down-icon';
        thumbsDownIcon.addEventListener('click', () => {
            // No functionality for thumbs down for now
        });

        message.appendChild(thumbsUpIcon);
        message.appendChild(thumbsDownIcon);
    });
}

// Call the function to add thumbs up and thumbs down icons to commentary messages
addThumbsIconsToCommentary();

// Example usage
backend.sendMessage('left-block', 'Legislation block is here!');
backend.sendMessage('right-block', 'Commentary block is here!');
backend.sendMessage('right-block', 'Another message in commentary block!');
backend.sendMessage('left-block', 'Another message in legislation block!');