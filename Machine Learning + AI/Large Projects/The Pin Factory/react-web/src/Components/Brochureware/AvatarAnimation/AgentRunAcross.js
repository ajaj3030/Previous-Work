import React, { useState, useEffect, useContext } from 'react';
import { AgentContext } from './AgentRunAcrossController';
import './Avatar.css';

import frame0 from '../../../assets/baby-animation/right/right-0.png';
import frame1 from '../../../assets/baby-animation/right/right-1.png';
import frame2 from '../../../assets/baby-animation/right/right-2.png';
import frame3 from '../../../assets/baby-animation/right/right-3.png';

const frames = [frame0, frame1, frame2, frame3];

const AgentRunAcross = ({ agent, screenWidthOffset }) => {
  // Set up state using the useState hook
  const [currentFrame, setCurrentFrame] = useState(0);
  const { updateAgentPosition } = useContext(AgentContext); // Get updateAvatar from context
  const [x, setX] = useState(agent.x);

  const screenWidth = window.innerWidth;
  const threshold = screenWidth + screenWidthOffset;

  useEffect(() => {
    const frameDelay = 120; // Specify the delay between frames in milliseconds
    let frameCounter = 0;
    let animateTimeout; // Declare animateTimeout
  
    const animateSprite = () => {
      setCurrentFrame(frameCounter);
      frameCounter = (frameCounter + 1) % 4;
      animateTimeout = setTimeout(animateSprite, frameDelay);
    };
  
    animateSprite(); // You need to call animateSprite to start the animation
  
    return () => clearTimeout(animateTimeout); // Now animateTimeout is defined
    
  }, []);

  useEffect(() => {
    const moveSprite = () => {
      setX((prevX) => {
        const newX = prevX + 5;
        if (newX >= threshold) {
          clearTimeout(moveTimeout);
        }
        updateAgentPosition(agent.id, newX); // Update the avatar's x value in the parent component
        return newX;
      });
    };
    let moveTimeout = setTimeout(moveSprite, 20);
    return () => clearTimeout(moveTimeout);
  }, [x, screenWidthOffset, agent.id, updateAgentPosition, threshold])

  return (
    <img src={frames[currentFrame]} alt="Avatar Frame" className="h-12 w-auto max-w-full object-contain absolute" style={{ left: `${agent.x}px`, top: `${agent.y}px` }}/>
  );
};

export default AgentRunAcross ;