import React, { useState, useContext, useRef, useEffect } from 'react';
import StatusGraphic from './StatusGraphic';
import { PlusIcon, ArrowLeftIcon } from '@heroicons/react/20/solid'
import { Link } from 'react-router-dom';

import { ModelContext } from './ModelContext';


export default function AllAgentsPage( {onNext, onPrev, fromLocation }) {
    
    const [values, setValues] = useState({});
    const [formValid, setFormValid] = useState(false);
    const [formSubmitted, setFormSubmitted] = useState(false);
    const [submittedSuccessfully, setSubmittedSuccessfully] = useState(false);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const checkFormValidity = () => {
        // Check if the required fields are filled
        const requiredFields = ["first_name", "last_name", "email"];
        const isFormValid = requiredFields.every(field => !!values[field]);
        setFormValid(isFormValid);
    };

    const handleInputChange = (event) => {
        const { name, value } = event.target;
        setValues(prevValues => ({...prevValues, [name]: value }));
        checkFormValidity();
    };
    
    const { state, setState } = useContext(ModelContext);
    const handleSubmit = async (e) => {
        e.preventDefault();
        // onNext();
    };

    const handleAgentClick = async (e, agent) => {
        e.preventDefault();
        setState({ ...state,
            selectedAgent: agent
          });
        onNext();
    };

    const [avatarIdHovered, setAvatarIdHovered] = useState(null);

    const handleBackClick = () => {
        onPrev();
    };

    const ref = useRef(null);
    const [width, setWidth] = useState(null);
    const [height, setHeight] = useState(null);

    useEffect(() => {
        setWidth(ref.current.offsetWidth);
        setHeight(ref.current.offsetHeight);
    }, []);

    return (
        <form action="#" method="POST" className="relative z-10 my-12 w-full mx-24" onSubmit={handleSubmit}>
            
            <div className='flex items-center justify-between mb-6'>
                <button onClick={handleBackClick}><ArrowLeftIcon className='w-6 text-gray-900 hover:opacity-70'/></button>
            </div>

            {/* Content Container */}
            <div className="mb-16">
                {/* Header */}
                <div className='mb-16'>
                    <h2 className="text-3xl font-bold tracking-tight text-gray-900">Say hello to your new agents</h2>
                    <div>
                        <p className="mt-2 text-md leading-6 text-gray-600">
                            Below is a summary of the agents you've decided to create today. You can always add more later.
                        </p>
                    </div>
                </div>

                {/* Agent grid */}
                <div className='flex items-center justify-center'>
                    <div className="grid grid-cols-3 gap-x-8 gap-y-8">
                        {state.agents.map((agent) => (
                            <button 
                                ref={ref} 
                                onClick={(e) => handleAgentClick(e, agent)}
                                onMouseEnter={() => setAvatarIdHovered(agent.id)}
                                onMouseLeave={() => setAvatarIdHovered(null)}
                                className='flex flex-col items-center rounded-md hover:bg-gray-50 py-4 px-8'
                                >
                                <div
                                    className='flex items-center justify-center'
                                >
                                    {avatarIdHovered == agent.id ? (
                                        <div className='mt-1'>{agent.avatar.animated}</div>
                                    ) : (
                                        <div className='mt-1'>{agent.avatar.static}</div>
                                    )}
                                </div>
                                <p className='mt-3 font-semibold text-gray-900'>{agent.name}</p>
                                <p className='text-xs font-sembiold text-gray-500'>{agent.model.name}</p>
                                <div className='mt-3 flex items-center justify-center mt-2'>
                                    <StatusGraphic agent={agent} />
                                </div>
                            </button>
                        ))}
                        <button
                        onClick={onPrev}
                        className={`flex flex-col items-center justify-center rounded-md hover:outline-1.5 hover:outline-gray-300 hover:outline-dashed py-4 px-8`}
                        style={{ width: `${width}px`, height: `${height}px` }}
                        >
                            <PlusIcon className='h-5 w-5 text-gray-500'/>
                            <p className='mt-2 text-sm font-semibold text-gray-500'>Add Agent</p>
                        </button>
                    </div>
                </div>

            </div>

            {/* Button container */}
            <Link to="/dashboard" >
                <div className='flex items-center justify-center'>
                    <button className={`ml-1 block w-full rounded-md px-3.5 py-2.5 text-center text-sm font-semibold text-white shadow-sm focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 bg-blue-900 hover:bg-blue-900/60`}>
                        <div className="flex items-center justify-center">
                            Add agents to your factory
                            {/* Loading SVG below can go here */}
                        </div>
                    </button>
                </div>
            </Link>
            
        </form>
    );
};

// <button className='flex flex-col items-center rounded-md hover:bg-gray-100 py-4 px-8'>
{/* <AvatarWalkingForwards stat={"still"}/>
<p className='mt-2 font-semibold text-gray-900'>Colin</p>
<p className='-m-0.5 text-xs font-sembiold text-gray-500'>GPT-4</p>
<p className='mt-2 text-xs font-bold text-emerald-700'>Completed</p>
</button> */}