import React, { useState, useEffect } from 'react';
import frame0 from '../../../assets/baby-animation/down/down-0.png';
import frame1 from '../../../assets/baby-animation/down/down-1.png';
import frame2 from '../../../assets/baby-animation/down/down-2.png';
import frame3 from '../../../assets/baby-animation/down/down-3.png';

import v_frame0 from '../../../assets/baby-animation/down/down-0.png';
import v_frame1 from '../../../assets/baby-animation/down/down-1.png';
import v_frame2 from '../../../assets/baby-animation/down/down-2.png';
import v_frame3 from '../../../assets/baby-animation/down/down-3.png';

const frames = [frame0, frame1, frame2, frame3];
const v_frames = [v_frame0, v_frame1, v_frame2, v_frame3];

export const AvatarWalkingForwards = ({status, avatar}) => {
  // Set up state using the useState hook
  const [currentFrame, setCurrentFrame] = useState(0);

  useEffect(() => {

    const frameDelay = 150; // Specify the delay between frames in milliseconds
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
  return (
    <div>
      {status === "forwards" ? (
        <div><img src={avatar === "VODAFONE" ? v_frames[currentFrame] : frames[currentFrame]} alt="Avatar Frame" className="h-14"/></div>
      ) : (
        <div><img src={avatar === "VODAFONE" ? v_frames[0] : frames[0]} alt="Avatar Frame" className="h-14"/></div>
      )}
    </div>
  );
};