import React, { useState, useEffect, useContext, useCallback } from 'react';
import { AgentContext } from './AgentFactoryController';
import './Avatar.css';



// RENDERS THE AVATAR ANIMATION FOR AN INDIVIDUAL AVATAR
// ----------------------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------------------

const AvatarAnimation = ({ agent, containerWidth, containerHeight, avatarSize=12, agentIsClicked }) => {
  
  // VARIABLES
  // --------------------------------------------------------

  // Avatar moving state
  const [isAvatarMoving, setIsAvatarMoving] = useState(Math.random() > 0.5); // Randomly initialised.

  // Avatar position variables
  const { agents, updateAgentPosition } = useContext(AgentContext); // Get updateAgentPosition from context

  const [x, setX] = useState(agent.x); // Avatar's x position
  const [y, setY] = useState(agent.y); // Avatar's y position

  // Avatar direction variables
  const [direction, setDirection] = useState(agent.avatar.directions[Math.floor(Math.random() * agent.avatar.directions.length)]); // Randomly initialised.

  // Avatar frame-to-frame variables
  const [currentFrame, setCurrentFrame] = useState(0); // Current frame of the animation
  const number0fFrames = 4; // Number of frames in the animation 

  // Avatar time and probability variables
  const stopProbability = 0.003; // Adjust this value to control how often the avatar stops randomly
  const minStopTime = 1000;  // The minimum time the avatar will stop for (in ms)
  const maxStopTime = 8000;  // The maximum time the avatar will stop for (in ms)

  const probChangesDirectionSecondTimeAfterStopping = 0.3; // Probability of changing direction second time after stopping
  const changeDirectionAt_X_OfStartMovingStartTime = 0.8; // Adjust this value to control how often the avatar changes direction

  // The padding around the whole container given to the avatar. It will hit the borders of the container including this padding.
  const containerPadding = 100;

  // Avatar speed variables
  const [pixelIncrement, setPixelIncrement] = useState({ 'LEFT': 1.5, 'RIGHT': 1.5, 'UP': 1, 'DOWN': 1.5 });
  const [positionFrequency, setPositionFrequency] = useState({ 'LEFT': 20, 'RIGHT': 20, 'UP': 20, 'DOWN': 20 });
  const [frameSpeed, setFrameSpeed] = useState(150); // Specify the delay between frames in milliseconds


  // ANIMATION
  // --------------------------------------------------------

  // Starts avatar moving again when it is still (either at stop/border or at animation init)
  useEffect(() => {
    if (!isAvatarMoving) {
      // Randomise the stop time between minStopTime and maxStopTime
      const startTime = Math.random() * (maxStopTime - minStopTime) + minStopTime;
      // If changes direction second time after stopping, change direction after 80% of stop time
      if (Math.random() < probChangesDirectionSecondTimeAfterStopping) {
        setTimeout(() =>
          changeDirection(), (startTime*changeDirectionAt_X_OfStartMovingStartTime)
        );
      }
      // Change direction after randomised start time
      setTimeout(() =>
        setIsAvatarMoving(true), startTime
      );
    }
  }, [isAvatarMoving]);

  // Runs when avatar is moving (every 1px). Probability of avatar stopping is determined by stopProbability.
  useEffect(() => {
    // Stops moving depending on stop probability
    if (Math.random() < stopProbability && !agentIsClicked) {
      // Stop moving
      setIsAvatarMoving(false);
      // Randomise the length of the stop time between minStopTime and maxStopTime
      const stopTime = Math.random() * (maxStopTime - minStopTime) + minStopTime;
      // Change direction after 80% of stop time
      setTimeout(() =>
        changeDirection(), (stopTime*changeDirectionAt_X_OfStartMovingStartTime)
      );
      // Start moving again after randomised stop time
      setTimeout(() =>
        setIsAvatarMoving(true), stopTime
      );
    }
  }, [x, y]);

  // Sets the frame animation. Runs whenever isAvatarMoving changes (at stop/start and border collision).
  useEffect(() => {
    // If effect triggered on avatar stop moving
    if (!isAvatarMoving) {
      // Set frame to "Still" animation frame 
      setCurrentFrame(0);
      return;
    } else { // If effect triggered on avatar start moving
      // Set frame to first animation frame
      let frameCounter = 0;
      let animateTimeout;
      // Animate sprite function
      const animateSprite = () => {
        // Update currentFrame state with frameCounter
        setCurrentFrame(frameCounter);
        // Increment frame counter (returns to 0 when reaches number0fFrames)
        frameCounter = (frameCounter === (number0fFrames-1)) ? 0 : (frameCounter + 1);
        // Set frame to change every frameSpeed milliseconds
        animateTimeout = setTimeout(animateSprite, frameSpeed);
      };
      // Run animate sprite function
      animateSprite();
      // Clear timeout when component unmounts
      return () => clearTimeout(animateTimeout);
    }
  }, [isAvatarMoving, frameSpeed]);

  useEffect(() => {
    // Set frame to first animation frame
    if (agentIsClicked) {
      // Increase speed of agent
      setFrameSpeed(50);
      setPixelIncrement({ 'LEFT': 1, 'RIGHT': 1, 'UP': 1, 'DOWN': 1 }); // how much agents move each call - higher = faste
      setPositionFrequency({ 'LEFT': 3, 'RIGHT': 3, 'UP': 3, 'DOWN': 3 }); // lower this is, faster movement gets called - lower=faster
    } else {
      setPixelIncrement({ 'LEFT': 1.5, 'RIGHT': 1.5, 'UP': 1, 'DOWN': 1.5 });
      setPositionFrequency({ 'LEFT': 20, 'RIGHT': 20, 'UP': 20, 'DOWN': 20 });
      setFrameSpeed(150);
    }
  }, [agentIsClicked]);

  // Function that makes the avatar move (the absolute position on the scren). Called within the useEffect below.
  const moveAvatar = () => {
    if (!isAvatarMoving) { // If avatar is not moving, return.
      return;
    } else { // If avatar is moving...
      if (direction === 'RIGHT') {
        setX((old_x) => {
          // 'new_x' position is the 'old_x' plus the pixelIncrement 'right' -- (moving x-positive/right).
          const new_x = old_x + pixelIncrement[direction];

          if ((containerWidth != null && new_x >= containerWidth - containerPadding)) { // If out of bounds...
            // First stop moving
            if (isAvatarMoving) {
              setIsAvatarMoving(false);
            };
            // Then change direction
            changeDirection();
          } else { // If not out of bounds...
            // Update the avatar's x value in the parent component. Re-triggers useEffect.
            updateAgentPosition(agent.avatar.id, new_x, y);
          }
          return new_x;
        });
      }
      if (direction === 'LEFT') {
        setX((old_x) => {
          // 'new_x' position is the 'old_x; minus the pixelIncrement for 'left' -- (moving x-negative/left).
          const new_x = old_x - pixelIncrement[direction];
          if ((containerWidth != null && new_x <= 0 + containerPadding)) { // If out of bounds...
            // First stop moving
            if (isAvatarMoving) {
              setIsAvatarMoving(false);
            };
            // Then change direction
            changeDirection();
          } else { // If not out of bounds...
            // Update the avatar's x value in the parent component. Re-triggers useEffect.
            updateAgentPosition(agent.avatar.id, new_x, y);
          }
          return new_x;
        });
      }
      if (direction === 'UP') {
        setY((old_y) => {
          // 'new_y' position is the 'old_y' minus the pixelIncrement for 'up' -- (moving y-negative/up).
          const new_y = old_y - pixelIncrement[direction];
          if ((containerHeight != null && new_y <= 0 + containerPadding)) { // If out of bounds...
            // First stop moving
            if (isAvatarMoving) {
              setIsAvatarMoving(false);
            };
            // Then change direction
            changeDirection();
          } else { // If not out of bounds...
            // Update the avatar's y value in the parent component. Re-triggers useEffect.
            updateAgentPosition(agent.avatar.id, x, new_y);
          }
          return new_y;
        });
      }
      if (direction === 'DOWN') {
        setY((old_y) => {
          // 'new_y' position is the 'old_y' plus the pixelIncrement for 'down' -- (moving y-positive/down).
          const new_y = old_y + pixelIncrement[direction];
          if ((containerHeight != null && new_y >= containerHeight - containerPadding)) { // If out of bounds...
            // First stop moving
            if (isAvatarMoving) {
              setIsAvatarMoving(false);
            };
            // Then change direction
            changeDirection();
          } else { // If not out of bounds...
            // Update the avatar's y value in the parent component. Re-triggers useEffect.
            updateAgentPosition(agent.avatar.id, x, new_y);
          }
          return new_y;
        });
      }
    }
  };

  // calls moveAvatar function every. Runs either when avatar is moving (updateAgentPosition) or when avatar starts moving (isAvatarMoving).
  useEffect(() => {
    // Run moveAvatar function every positionFrequency milliseconds
    let moveTimeout = setTimeout(moveAvatar, positionFrequency[direction]);
    // Clear timeout when component unmounts
    return () => clearTimeout(moveTimeout);
  }, [updateAgentPosition, isAvatarMoving])

  // Function that returns the probability of the next direction based on the avatar's position
  function calculateDirectionProbabilities() {
    const left = x/containerWidth;
    const right = 1-left;
    const up = y/containerHeight;
    const down = 1-up;
    const sum = left+right+up+down;
    var unorderedWeightings = {};
    unorderedWeightings["LEFT"] = roundToDecimalPlaces((left/sum), 2);
    unorderedWeightings["RIGHT"] = roundToDecimalPlaces((right/sum), 2);
    unorderedWeightings["UP"] = roundToDecimalPlaces((up/sum), 2);
    unorderedWeightings["DOWN"] = roundToDecimalPlaces((down/sum), 2);
    let weightings = new Map(Object.entries(unorderedWeightings).sort((a, b) => b[1] - a[1]));
    return weightings;
}

  // Function that changes the direction of the avatar when randomly triggered
  function changeDirection() {
    // Choose a random direction based on the weights
    const randomNum = Math.random();
    let cumulativeProbability = 0;
    const directionWeights = calculateDirectionProbabilities();

    // Loop through the direction weights
    for (const [direction, probability] of directionWeights) {
        // Add the probability to the cumulative probability
        cumulativeProbability += probability;
        // If the random number is less than the cumulative probability, choose that direction
        if (randomNum <= cumulativeProbability) {
          // Set the direction to the chosen direction
          setDirection(direction);
          return;
        }
    };
    
    const nextDirection = agent.avatar.directions[Math.floor(Math.random()*3)];
    setDirection(nextDirection);
    return;
  };

  // Avatar Render
  return (
    // Dynamic image source based on currentFrame, absolute position based on x and y updates
    <img src={agent.avatar.frames[direction][currentFrame]} alt="Avatar Frame" className={`h-${avatarSize} w-auto max-w-full object-contain absolute`} style={{ left: `${x}px`, top: `${y}px` }}/>
    
  );
};


export default AvatarAnimation;

function roundToDecimalPlaces(number, decimals) {
  return parseFloat(number.toFixed(decimals));
}