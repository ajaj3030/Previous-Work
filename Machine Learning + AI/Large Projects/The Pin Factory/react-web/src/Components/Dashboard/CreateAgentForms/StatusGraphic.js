import React from 'react';
import { capitalizeFirstLetter } from './FullAgentPage';

export function areAllStepsCompleted(agent) {
    console.log("areAllStepsCompleted() triggered")
    if (Object.values(agent.setup).every(step_status => step_status === 'COMPLETED')) {
        return 'COMPLETED';
    } else if (Object.values(agent.setup).some(step_status => (step_status === 'IN PROGRESS' || step_status === 'COMPLETED'))) {
        return 'IN PROGRESS';
    } else {
        return 'NOT STARTED';
    }
}

export default function StatusGraphic({ agent, isBubbleBackground=true }) {
    const status = areAllStepsCompleted(agent);
    if (status === 'IN PROGRESS') {
        return (
            <span className={`text-[11px] font-bold text-purple-700 ${!isBubbleBackground ? "" : "inline-flex items-center rounded-md px-1.5 py-0.5 font-bold bg-purple-500/[0.15]"}`}>
                {capitalizeFirstLetter(status)}
            </span>
        );
    } else if (status === 'COMPLETED') {
        return (
            <span className={`text-[11px] font-bold text-green-700 ${!isBubbleBackground ? "" : "inline-flex items-center rounded-md px-1.5 py-0.5 bg-green-500/[0.15]"}`}>
                {capitalizeFirstLetter(status)}
            </span>
        );
    } else {
        return (
            <span className={`text-[11px] font-bold text-orange-800 ${!isBubbleBackground ? "" : "inline-flex items-center rounded-md px-1.5 py-0.5 bg-orange-400/[0.2]"}`}>
                {capitalizeFirstLetter(status)}
            </span>
        );
    }
};