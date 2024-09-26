import React, { createContext, useState } from 'react';

export const TaskContext = createContext();

export const TaskProvider = ({ children }) => {
  const [activeTask, setActiveTask] = useState(null);

  return (
    <TaskContext.Provider value={{ activeTask, setActiveTask }}>
      {children}
    </TaskContext.Provider>
  );
};
