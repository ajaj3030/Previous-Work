
import React from 'react';
import { useParams } from 'react-router-dom';

export default function AgentPage() {

  let { agentId } = useParams();

  return (
    <div>
      <p>Agent ID: {agentId}</p>
    </div>
  )
}
