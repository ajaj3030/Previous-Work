
onmessage = function(event) {
  const { avatarHeight, screenHeight, agents, avatarWidth, spacing, brochurewareMargin, yMarginOn, yMargin } = event.data;
  let newY;

  const initialX = 0 - avatarWidth - spacing;

  // Generate a random y position within the screen height
  const generateRandomY = () => Math.floor(Math.random() * (screenHeight - avatarHeight));

  // Define a function that checks if a y position is valid (i.e., it doesn't result in overlapping agents)
  const isValidY = (y) => {
    if (yMarginOn && y < yMargin) {
      return false;
    } else {
      return !agents.some(avatar => {
        return y >= avatar.y && y <= avatar.y + avatar.height + spacing && initialX >= avatar.x && initialX <= avatar.x + avatar.width + spacing;
      });
    }
  };

  // Generate a valid y position for the new avatar
  do {
      newY = generateRandomY();
  } while (!isValidY(newY));
  // Post the new x and y values to the main thread
  self.postMessage({ newY: newY, newX: initialX });
};
