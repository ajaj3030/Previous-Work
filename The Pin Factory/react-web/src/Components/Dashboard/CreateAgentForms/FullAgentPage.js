import React, { useContext, useState, useEffect } from 'react';
import { ModelContext } from './ModelContext';
import { ArrowLeftIcon } from '@heroicons/react/20/solid'
import { ArrowStalk, ArrowHead } from '../../Brochureware/Header/LogInButton.js';
import StatusGraphic from './StatusGraphic.js';
import { areAllStepsCompleted } from './StatusGraphic.js';
import { allSections } from './eachSectionInfo.js';


const status_description = {
    'NOT STARTED': "You have not started {AGENT}'s setup. Complete the steps below to build {AGENT}'s brain and unlock {PRONOUN_POSSESSIVE} benefits.",
    'IN PROGRESS': "Finish the steps below to complete {AGENT}'s setup, and put {PRONOUN_OBJECTIVE} to work.",
    'COMPLETED': "Great work! You have completed {AGENT}'s setup. Add {AGENT} to your factory to start working with {PRONOUN_OBJECTIVE}.",
}

function classNames(...classes) {
    return classes.filter(Boolean).join(' ')
}

export function capitalizeFirstLetter(string) {
    return string.charAt(0).toUpperCase() + string.slice(1).toLowerCase();
}


export default function FullAgentPage( {onNext, onPrev, modelId, toggleExpand }) {

    const { state, setState } = useContext(ModelContext);
    const handleBackClick = () => {
        onPrev();
        console.log(state);
    };

    return (
        <div className="relative z-10 my-12 w-full mx-24">
            <div className='flex items-center justify-between mb-6'>
                <button onClick={handleBackClick}><ArrowLeftIcon className='w-6 text-gray-900 hover:opacity-70'/></button>
            </div>
            <div className='flex items-center justify-between mb-6'>
                <div className='flex items-center'>
                    <AvatarCircle />
                    <div className='ml-6 place-items-start'>
                        <h2 className="text-xl font-bold tracking-tight text-gray-900">{state.selectedAgent.name}</h2>
                        <div className='flex items-center'>
                            <ModelBubble />
                            <StatusBubble />
                        </div>
                    </div>
                </div>
            </div>
            <SetupStepsButtons toggleExpand={toggleExpand} />
        </div>
    );
};


function AvatarCircle({}) {
    const { state, setState } = useContext(ModelContext);
    const [avatarHovered, setAvatarHovered] = useState(false);
    return (
        <div
            onMouseEnter={() => setAvatarHovered(true)}
            onMouseLeave={() => setAvatarHovered(false)}
            className='w-20 h-20 rounded-full border border-gray-300 bg-gray-100 flex items-center justify-center'
        >
            {avatarHovered ? (
                <div className='mt-1'>{state.selectedAgent.avatar.animated}</div>
            ) : (
                <div className='mt-1'>{state.selectedAgent.avatar.static}</div>
            )}
        </div>
    );
};

function ModelBubble({}) {

    const { state, setState } = useContext(ModelContext);
    const [modelBubbleIsOpen, setModelBubbleIsOpen] = useState(false);
    const [modelBubbleIsTransitioning, setModelBubbleIsTransitioning] = useState(false);

    useEffect(() => {
        if (modelBubbleIsTransitioning) setModelBubbleIsTransitioning(true);
    }, [modelBubbleIsTransitioning]);

    return (
        <div 
        onMouseEnter={() => setModelBubbleIsOpen(true)}
        onMouseLeave={() => setModelBubbleIsOpen(false)}
        className={`relative cursor-default`}
        >
            <span className="inline-flex items-center rounded-md px-1.5 py-0.5 text-[11px] font-bold bg-gray-500/[0.15] text-gray-500">
                {state.selectedAgent.model.name}
            </span>
            {modelBubbleIsOpen || modelBubbleIsTransitioning ? (
                <div
                    className={`absolute z-10 pt-2 -ml-6 flex max-w-min ${
                    modelBubbleIsOpen
                    ? 'opacity-100 transition-opacity duration-300 ease-out'
                    : 'opacity-0 transition-opacity duration-150 ease-in'
                    } ${(modelBubbleIsTransitioning || modelBubbleIsOpen) ? 'transitioning' : ''}`}
                    onTransitionEnd={() => setModelBubbleIsTransitioning(false)}
                >
                    <div className={`w-56 shrink rounded-md bg-white px-3 py-2 text-xs font-semibold leading-4 text-gray-600 shadow-lg ring-1 ring-gray-900/5`}>
                        <p>{state.selectedAgent.model.description}</p>
                    </div>
                </div>
            ) : null}
        </div>
    );
};

function StatusBubble({}) {

    const { state, setState } = useContext(ModelContext);
    const [statusBubbleIsOpen, setStatusBubbleIsOpen] = useState(false);
    const [statusBubbleIsTransitioning, setStatusBubbleIsTransitioning] = useState(false);
    
    useEffect(() => {
        if (statusBubbleIsOpen) setStatusBubbleIsTransitioning(true);
    }, [statusBubbleIsOpen]);

    const [setupStatus, setSetupStatus] = useState(areAllStepsCompleted(state.selectedAgent));

    useEffect(() => {
        setSetupStatus(areAllStepsCompleted(state.selectedAgent));
    }, [state.selectedAgent]);

    return (
        <div 
        onMouseEnter={() => setStatusBubbleIsOpen(true)}
        onMouseLeave={() => setStatusBubbleIsOpen(false)}
        className={`relative ml-1 cursor-default`}
        >
            <StatusGraphic agent={state.selectedAgent} />
            {statusBubbleIsOpen || statusBubbleIsTransitioning ? (
                <div
                    className={`absolute z-10 pt-2 -ml-6 flex max-w-min ${
                    statusBubbleIsOpen
                    ? 'opacity-100 transition-opacity duration-300 ease-out'
                    : 'opacity-0 transition-opacity duration-150 ease-in'
                    } ${(statusBubbleIsTransitioning || statusBubbleIsOpen) ? 'transitioning' : ''}`}
                    onTransitionEnd={() => setStatusBubbleIsTransitioning(false)}
                >
                    <div className={`w-56 shrink rounded-md bg-white px-3 py-2 text-xs font-semibold leading-4 text-gray-600 shadow-lg ring-1 ring-gray-900/5`}>
                        <p>
                            {
                                status_description[setupStatus]
                                    .replace(new RegExp('{AGENT}', 'g'), state.selectedAgent.name)
                                    .replace(new RegExp('{PRONOUN_POSSESSIVE}', 'g'), state.selectedAgent.avatar.sex.pronouns.possessive)
                                    .replace(new RegExp('{PRONOUN_OBJECTIVE}', 'g'), state.selectedAgent.avatar.sex.pronouns.objective)
                            }
                        </p>
                    </div>
                </div>
            ) : null}
        </div>
    );
};


function SetupStepsButtons({toggleExpand}) {

    const { state, setState } = useContext(ModelContext);
    const [stepHovered, setStepHovered] = useState(0);
    const handleAddClick = (selected_section) => {
        toggleExpand(selected_section);
    };

    return (
        <div className='flex flex-col divide-y divide-gray-200'>
            {allSections.map((step) => (
                <button 
                    onClick={() => handleAddClick(step.index)}
                    className="flex items-center justify-between py-5 focus:outline-none group hover:opacity-70"
                    onMouseEnter={() => setStepHovered(step.index)}
                    onMouseLeave={() => setStepHovered(0)}
                >
                    <div className="flex gap-x-4">
                        <step.icon className={`mt-1 ${step.name == "training" ? "h-[24px]" : "h-[19px]"} text-gray-500`} />
                        <div className="min-w-0 flex-auto text-start">
                            <p className={`text-sm font-semibold leading-6 text-gray-900`}>{capitalizeFirstLetter(step.name)}</p>
                            <p className={`mt-1 mb-2 text-xs leading-4.5 text-gray-500`}>{step.description}</p>
                            <div className="flex items-center">
                                {state.selectedAgent.setup[step.name] === "IN PROGRESS" ? (
                                    <div className={`h-1 w-1 bg-purple-800 rounded-full mr-1`}></div>
                                ) : null}
                                <div>
                                    <p
                                        className={`text-xs font-bold ${state.selectedAgent.setup[step.name] === "COMPLETED" ? "text-green-700" : (state.selectedAgent.setup[step.name] === "NOT STARTED" ? "text-orange-800" : "text-purple-800")} ${stepHovered === step.index ? "opacity-70" : "opacity-100"}`}
                                    >
                                        {capitalizeFirstLetter(state.selectedAgent.setup[step.name])}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                    {/* <ChevronRightIcon className="h-5 w-5 text-gray-400" aria-hidden="true" /> */}
                    <div className="flex items-center">
                        <ArrowStalk className={`w-3 h-5 group-hover:scale-x-150`} />
                        <ArrowHead className={`w-2 h-3 -ml-1.5 group-hover:translate-x-0.5`} />
                    </div>
                </button>
            ))}
        </div>
    );
};