import React, { useState, useContext } from 'react';
import { XMarkIcon, ChevronDownIcon } from '@heroicons/react/20/solid'

import { getSectionDescription, getSectionBullets } from './eachSectionInfo';
import { ModelContext } from './ModelContext';
    
export default function AddMemory({ toggleExpand }) {

    const { state, setState } = useContext(ModelContext);

    const indexString = "memory";

    function updateAgentMemoryStatus(newStatus) {
        // Update the agent's setup status in the agents array
        const updateAgents = state.agents.map(agent => {
            if(agent.id === state.selectedAgent.id) {
                return {
                    ...agent,
                    setup: {
                        ...agent.setup,
                        memory: newStatus,
                    }
                }
            }
            return agent;
        });
        // Update the selected agent's setup status in the selectedAgent object
        const updateSelectedAgent = { 
            ...state.selectedAgent, 
            setup: {
                ...state.selectedAgent.setup,
                memory: newStatus,
            } 
        };
        // Write updates to the 'state' context
        setState({
            ...state,
            agents: updateAgents,
            selectedAgent: updateSelectedAgent,
        });

    }
    

    const handleMinimise = () => {
        toggleExpand();
    };

    const [whyIsExpanded, setWhyIsExpanded] = useState(false);

    function splitForItalicise(text, words_to_italicise) {
        // Split string around each word in the words_to_italicise array
        const parts = text.split(new RegExp(`(${words_to_italicise.join('|')})`, 'g'));
    
        return parts.map((part, i) => (
            words_to_italicise.includes(part)
            ? <span key={i} className="italic">{part}</span>
            : part
        ));
    }
    

    return (
        
        <div className="relative z-10 my-12 w-full mx-24 flex flex-col justify-between">

            {/* HEADER */}
            <div>
                <div className="border-b border-gray-200 pb-2">
                    {/* TITLE AND BACK BUTTON */}
                    <div className="flex items-center justify-between">
                        <h1 id="message-heading" className="text-base font-semibold leading-6 text-gray-900">
                            Memory
                        </h1>
                        <button onClick={handleMinimise}><XMarkIcon className='w-6 text-gray-900'/></button>
                    </div>
                    {/* MINI DESC */}
                    <h1 id="memory-description" className="mt-2 text-sm leading-4.5 text-gray-500">
                        {getSectionDescription(indexString)}
                    </h1>
                </div>

                <div className="mt-3 ring-1 ring-gray-200 bg-gray-200/50 shadow-sm rounded-md py-1.5 px-2.5">
                    <button onClick={() => setWhyIsExpanded(!whyIsExpanded)} className='w-full'>
                        <div className="flex items-center justify-between">
                            <p className={`ml-2 text-sm font-semibold text-gray-600`}>What is memory and why should I use it?</p>
                            <ChevronDownIcon className={`w-5 text-gray-600 transition-all duration-300 ease-in-out ${whyIsExpanded ? 'rotate-180' : '0'}`}/>
                        </div>
                    </button>
                    <div className={`overflow-hidden transition-all duration-300 ease-in-out ${whyIsExpanded ? 'max-h-96' : 'max-h-0'}`}>
                        <ul role="list" className="space-y-1 mt-1">
                            {getSectionBullets(indexString).map((bullet, bulletIdx) => (
                                <li key={bullet} className="relative flex gap-x-2">
                                    <div className="relative flex h-6 w-6 flex-none items-center justify-center">
                                        <div className="h-1 w-1 mt-0.5 rounded-full bg-gray-200 ring-1 ring-gray-300" />
                                    </div>
                                    <div className="text-sm py-0.5 leading-5 text-gray-600">
                                        {splitForItalicise(bullet, ["hallucinating", "remembers"])}
                                    </div>
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

            </div>

            {/* BODY */}
            <div className="flex items-center justify-center">
                <p className='text-sm font-semibold text-gray-800'>MEMORY CONTENT</p>
            </div>
            

            {/* Button Container */}
            <div className='flex items-center justify-between gap-x-3.5'>
                <button onClick={() => updateAgentMemoryStatus("NOT STARTED")} className={`mb-4 block w-full rounded-md bg-red-700/60 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:opacity-70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`}>
                    <div className="flex items-center justify-center">
                        Not started
                        {/* Loading SVG below can go here */}
                    </div>
                </button>
                <button onClick={() => updateAgentMemoryStatus("IN PROGRESS")} className={`mb-4 block w-full rounded-md bg-purple-700/60 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:opacity-70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`}>
                    <div className="flex items-center justify-center">
                        In progress
                        {/* Loading SVG below can go here */}
                    </div>
                </button>
                <button onClick={() => updateAgentMemoryStatus("COMPLETED")} className={`mb-4 block w-full rounded-md bg-green-700/60 px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm hover:opacity-70 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600`}>
                    <div className="flex items-center justify-center">
                        Completed
                        {/* Loading SVG below can go here */}
                    </div>
                </button>
            </div>
        </div>
    );
};