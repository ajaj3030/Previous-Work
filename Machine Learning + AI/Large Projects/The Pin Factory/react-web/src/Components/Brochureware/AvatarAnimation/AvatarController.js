import { useState, useEffect, useCallback, useMemo, useRef, createContext } from 'react';
import AvatarAnimation from './AvatarAnimation';
import { brochurewareMargin } from "../HomePage.js";
import { BasicAvatar, AllAvatars } from './Avatars';
export const AvatarContext = createContext();

function AvatarController({ agents_input=[], containerWidth=null, containerHeight=null }) {
  
  const initialX = -50;
  const initialY = (containerHeight/2);

  // Each avatar has an id, x and y
  const [agents, setAgents] = useState([]);

  // Update agents array
  useEffect(() => {
    console.log('Update agents array');
    // Take the agents_input array and create a new array of agents with x and y positions
    const new_agents_array = agents_input.map((agent, index) => {
      return {
        id: agent.id,
        x: initialX,
        y: initialY,
        avatar: AllAvatars[agent.avatar.id],
        isEntryAnimation: true,
        agent: agent
      }
    });
    setAgents(new_agents_array);
  }, [agents_input]);
  
  // It defines the updateAvatar function, which updates an avatar's x-y position
  function updateAgentPosition(id, new_x, new_y) {
      setAgents(old_agents => old_agents.map((agent) =>
      agent.id === id ? {...agent, x: new_x, y: new_y} : agent
    ));
  };


  // Render avatars
  return (
    <AvatarContext.Provider value={{ agents, updateAgentPosition }}>
      <div className='relative'>
        {agents.map(agent => (
          <AvatarAnimation
            agent={agent}
            containerWidth={containerWidth}
            containerHeight={containerHeight}
          />
        ))}
      </div>
    </AvatarContext.Provider>
  );
}

export default AvatarController;
