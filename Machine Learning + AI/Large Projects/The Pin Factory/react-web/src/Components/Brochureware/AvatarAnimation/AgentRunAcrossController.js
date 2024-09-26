import { useState, useEffect, useCallback, useMemo, useRef, createContext } from 'react';
import AgentRunAcross from './AgentRunAcross';
import { brochurewareMargin } from "../HomePage.js";

export const AgentContext = createContext();

function AgentRunAcrossController({ yMarginOn=false, yMargin=0, isStream=false, agentsRunning=true }) {
  // Each avatar has an id, x and y
  const [agents, setAgents] = useState([]);
  
  const [isRunning, setIsRunning] = useState(true);

  useEffect(() => {
    setIsRunning(agentsRunning);
  }, [agents]);

  // It sets the avatar height, screen height, avatar width, and spacing constant
  const avatarHeight = 50;
  const screenHeight = window.innerHeight;
  const avatarWidth = 50;
  const spacing = 10;

  const [isPageVisible, setIsPageVisible] = useState(!document.hidden);
  // This function will be called whenever the tab visibility changes
  const handleVisibilityChange = () => {
    setIsPageVisible(!document.hidden);
  };

  useEffect(() => {
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  // It creates a new web worker
  const worker = useMemo(() => new Worker(`${process.env.PUBLIC_URL}/AvatarWorker.js`), []);

  // It defines the addAvatar function, which sends a message to the web worker with the necessary data
  const addAvatar = useCallback(() => {
    worker.postMessage({ avatarHeight, screenHeight, agents, avatarWidth, spacing, brochurewareMargin, yMarginOn, yMargin });
  }, [agents, worker]);

  // It listens for a message from the web worker
  worker.onmessage = (event) => {
    // It gets the new x and y values from the web worker
    var { newY, newX } = event.data;
    if (isStream) {
      newY = 0;
    }
    // If the new x and y values are defined, it adds a new avatar with these values to the state
    if (newY !== undefined && newX !== undefined) {
      setAgents(agents => [
        ...agents,
        { 
          id: Math.random(),
          x: newX,
          y: newY,
          width: avatarWidth, // Remember to pass the width to your AvatarAnimation component
        }
      ]);
    }
  };

  // Values that determine the frequenc and randomness of avatar generation
  const randomness = 8;
  const frequency = 5;
  const timeoutId = useRef(null);

  useEffect(() => {
    // Clear any existing timeouts whenever frequency or randomness changes
    if (timeoutId.current) {
      clearTimeout(timeoutId.current);
      timeoutId.current = null;
    }
    console.log('isRunning: ' + isRunning)
    // Do not create new avatars if the page is not visible or if the isRunning prop is false
    if (!isPageVisible || !isRunning) {
      return; 
    }

    const interval = 5000 / frequency;

    const invokeAddAvatar = () => {
      if (isStream) {
        // Call addAvatar function every interval
        addAvatar();
        timeoutId.current = setTimeout(invokeAddAvatar, 3000);
      } else {
        addAvatar();
        const randFactor = Math.random()*(randomness / 10); // Adjust randomness factor as needed
        const delay = interval * (1 - randFactor);
        timeoutId.current = setTimeout(invokeAddAvatar, delay);
      }
    };

    // Kick off the first invocation
    invokeAddAvatar();

    // Clean up on unmount
    return () => {
      if (timeoutId.current) {
        clearTimeout(timeoutId.current);
        timeoutId.current = null;
      }
    };
  }, [frequency, randomness, isPageVisible, isRunning]);

  // It defines the updateAvatar function, which updates an avatar's x value
  const updateAgentPosition = useCallback((id, newX) => {
    setAgents(prevAgents => prevAgents.map((agent) =>
      agent.id === id ? {...agent, x: newX} : agent
    ));
  }, []);

  // Add a new useEffect hook to remove avatars that have moved off the screen
  useEffect(() => {
    const removeOffScreenAvatars = () => {
      const windowWidth = window.innerWidth;
      setAgents(agents => agents.filter(agent => agent.x <= windowWidth + 50));
    };

    const intervalId = setInterval(removeOffScreenAvatars, 100);
    return () => clearInterval(intervalId);
  }, []);

  // Render avatars
  return (
    <AgentContext.Provider value={{ agents, updateAgentPosition }}>
      <div className='relative'>
        {agents.map(agent => (
          <AgentRunAcross key={agent.id} agent={agent} screenWidthOffset={210} />
        ))}
      </div>
    </AgentContext.Provider>
  );
}

export default AgentRunAcrossController;