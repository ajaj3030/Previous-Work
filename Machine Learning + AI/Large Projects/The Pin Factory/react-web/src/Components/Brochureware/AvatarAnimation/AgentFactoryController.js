import { useState, useEffect, useCallback, useMemo, useRef, createContext } from 'react';
import AgentFactoryAnimation from './AgentFactoryAnimation';
import { BasicAvatar, AllAvatars } from './Avatars';
export const AgentContext = createContext();

function AgentFactoryController({ agents_input=[], containerWidth=null, containerHeight=null }) {
  
  const initialX = 100;
  const initialY = 300;

  // Each avatar has an id, x and y
  const [agents, setAgents] = useState([]);

  // Update agents array
  useEffect(() => {
    console.log('Update agents array');
    // Take the agents_input array and create a new array of agents with x and y positions
    const new_agents_array = agents_input.map((agent, index) => {
      return {
        id: index,
        x: initialX,
        y: initialY,
        avatar: BasicAvatar,
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

  const [agentIsClicked, setAgentIsClicked] = useState(false);

  // Speeds
  // function agentClicked() {
  //   setAgentIsClicked(true);
  //   setTimeout(function() {
  //     setAgentIsClicked(false)
  //   }, 600);
  // };

  useEffect(() => {
    const intervalId = setInterval(() => {
      // console.log('Agents:');
      // console.log(agents_input);
    }, 3000);

    // The return function is called when the component unmounts, clearing the interval
    return () => clearInterval(intervalId);
  }, []);
  

  // Render avatars
  return (
    <AgentContext.Provider value={{ agents, updateAgentPosition }}>
      <div className='relative'>
        {agents.map(agent => (
          <button
            // onClick={agentClicked}
          >
            <AgentFactoryAnimation
              key={agent.id}
              agent={agent}
              containerWidth={containerWidth}
              containerHeight={containerHeight}
              agentIsClicked={agentIsClicked}
            />
          </button>
        ))}
      </div>
    </AgentContext.Provider>
  );
}

export default AgentFactoryController;
