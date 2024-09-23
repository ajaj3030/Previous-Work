import React, { useState } from 'react';

const ModelContext = React.createContext();

function ModelProvider({ children }) {
  const [state, setState] = useState({
    agents: [],
    selectedAgent: null,
  });
  return (
    <ModelContext.Provider value={{ state, setState }}>
      {children}
    </ModelContext.Provider>
  );
}

export { ModelContext, ModelProvider }