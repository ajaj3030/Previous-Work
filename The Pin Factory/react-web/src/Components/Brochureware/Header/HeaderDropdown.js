import { React, Fragment, useState } from 'react';
import { Popover, Transition } from '@headlessui/react';
import { Link } from 'react-router-dom';

import {
    CubeTransparentIcon,
    Square3Stack3DIcon,
    ChatBubbleBottomCenterTextIcon,
    CloudArrowUpIcon,
    CodeBracketSquareIcon,
  } from '@heroicons/react/24/solid'


export function AgentsDropdown(props) {

    // For whole hover
    const [isOpen, setIsOpen] = useState(false);

    // For panel hover
    const [isHovered, setIsHovered] = useState(0);

    const handleMouseEnter = () => {
        setIsOpen(true);
    };
    
    const handleMouseLeave = () => {
        setIsOpen(false);
        setIsHovered(0);
    };

    return (
        <Popover className="relative">
            <div
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                className="relative items-center text-base font-semibold text-black relative"
            >
                <span className={isOpen ? 'opacity-50' : ''}>{props.name}</span>

                <Transition
                    show={isOpen}
                    as={Fragment}
                    enter="transition ease-out duration-200"
                    enterFrom="opacity-0 translate-y-1"
                    enterTo="opacity-100 translate-y-0"
                    leave="transition ease-in duration-150"
                    leaveFrom="opacity-100 translate-y-0"
                    leaveTo="opacity-0 translate-y-1"
                >
                    <Popover.Panel className={"absolute left-1/2 z-10 pt-5 flex w-screen max-w-max -translate-x-1/2 px-4"}>
                        <div className='relative'>
                            <PopoverHump className="absolute z-50" width={22} height={14} />
                            <div className="relative z-0 w-screen max-w-md flex-auto overflow-hidden rounded-md bg-white text-sm leading-6 drop-shadow-3xl ring-1 ring-gray-900/5">
                                <div className="p-4">
                                    {props.agents.map((item) => (
                                        <div
                                            key={item.name}
                                            className="group relative flex gap-x-6 rounded-md p-4 hover:bg-gray-50"
                                            onMouseEnter={() => setIsHovered(item.name)}
                                            >
                                        <div className="mt-1 flex h-11 w-11 flex-none items-center justify-center rounded-md">
                                            <AgentPng PNG={item.icon} size={(isHovered === item.name) ? 13 : 12} />
                                        </div>
                                        <div>
                                            <a href={item.href} className="font-semibold text-gray-900">
                                            {item.name}
                                            <span className="absolute inset-0" />
                                            </a>
                                            <p className="mt-1 text-gray-600">{item.description}</p>
                                        </div>
                                        </div>
                                    ))}
                                </div>
                                <div className={`grid grid-cols-1 divide-x divide-gray-900/5 bg-gray-50`}>
                                    {props.agentsBottom.map((item) => (
                                        <a
                                        key={item.name}
                                        href={item.href}
                                        className="flex items-center justify-center gap-x-2.5 p-3 font-semibold text-gray-900 hover:bg-gray-100"
                                        >
                                            <item.icon className="h-5 w-5 flex-none text-gray-400" aria-hidden="true" />
                                            {item.name}
                                        </a>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </Popover.Panel>
                </Transition>
            </div>
        </Popover>
    );
}

export function InfrastructureDropdown() {

    // For whole hover
    const [isOpen, setIsOpen] = useState(false);

    // For panel hover
    const [isHovered, setIsHovered] = useState(0);

    const memory = {
        name: 'Long-term agent memory',
        description: "Our agents retain information you upload. Whether that's an industrial database of financial reports or a small list of your favorite restaurants, our agents can remember it.",
        icon: CloudArrowUpIcon,
        isBeta: true,
    }

    const training = {
        name: 'Train agents without data',
        description: "With our intelligence protocol, you can train agents without training data. Just tell us what you want an agent to do and we'll do the rest.",
        icon: CubeTransparentIcon,
        isBeta: true,
    }

    const tasks = {
        name: 'Recursive task completion',
        description: "By maintaining task buckets and letting agents communicate, your factory of agents can complete and execute tasks that require multiple steps.",
        icon: Square3Stack3DIcon,
        isBeta: true,
    }

    const plugins = {
        name: 'Plug agents into the real world',
        description: "Authorise agents to execute tasks in your existing technical infrasturcture, and watch your workload fade away.",
        icon: CodeBracketSquareIcon,
        isBeta: true,
    }

    const handleMouseEnter = () => {
        setIsOpen(true);
    };
    
    const handleMouseLeave = () => {
        setIsOpen(false);
        setIsHovered(0);
    };

    return (
        <Popover className="relative">
            <div
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                className="relative items-center text-base font-semibold text-black relative"
            >
                <span className={isOpen ? 'opacity-50' : ''}>Infrastructure</span>

                <Transition
                    show={isOpen}
                    as={Fragment}
                    enter="transition ease-out duration-200"
                    enterFrom="opacity-0 translate-y-1"
                    enterTo="opacity-100 translate-y-0"
                    leave="transition ease-in duration-150"
                    leaveFrom="opacity-100 translate-y-0"
                    leaveTo="opacity-0 translate-y-1"
                >
                    <Popover.Panel className={"absolute left-1/2 z-10 pt-5 flex w-screen max-w-max -translate-x-1/2 px-4"}>
                        <div className='relative'>
                            <PopoverHump className="absolute z-50" width={22} height={14} />
                            <div className="relative z-0 flex-auto overflow-hidden rounded-md bg-white text-sm leading-6 drop-shadow-3xl ring-1 ring-gray-900/5">
                                <div className={`p-4 grid grid-cols-2`}>
                                    <Link
                                        to="/learn"
                                        className="flex items-start gap-x-2.5 mb-1.5 p-3 font-semibold text-gray-900 rounded-md hover:bg-gray-100"
                                        onMouseEnter={() => setIsHovered(memory.name)}
                                        >
                                            <div className='px-2 mt-1'>
                                                <memory.icon
                                                    className={`h-6 w-6 ${(isHovered === memory.name) ? "text-violet-600" : "text-gray-600"}`}
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div>
                                                <div>
                                                    {memory.name}
                                                    {memory.isBeta && (
                                                        <span className="ml-2 text-xs font-semibold bg-red-600/20 text-red-600 rounded-md px-1.5 py-0.5">
                                                            Alpha
                                                        </span>
                                                    )}
                                                </div>
                                                
                                                <div className='font-medium text-gray-500 leading-normal mt-1 max-w-sm'>
                                                    {memory.description}
                                                </div>

                                                <div className='mt-2.5'>
                                                    {/* Horizontal list of files accepted to upload each on gray background with rounded corners */}
                                                    <div className='flex gap-x-1.5'>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.pdf</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.docx</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.xls</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.pptx</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.csv</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === memory.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>.txt</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div> 
                                    </Link>
                                    <Link
                                        to="/learn"
                                        className="flex items-start gap-x-2.5 p-3 font-semibold text-gray-900 rounded-md hover:bg-gray-100 h-36"
                                        onMouseEnter={() => setIsHovered(training.name)}
                                        >
                                            <div className='px-2 mt-1'>
                                                <training.icon
                                                    className={`h-6 w-6 ${(isHovered === training.name) ? "text-violet-600" : "text-gray-600"}`}
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div>
                                                <div>
                                                    {training.name}
                                                    {training.isBeta && (
                                                        <span className="ml-2 text-xs font-semibold bg-red-600/20 text-red-600 rounded-md px-1.5 py-0.5">
                                                            Alpha
                                                        </span>
                                                    )}
                                                </div>
                                                
                                                <div className='font-medium text-gray-500 leading-normal mt-1 max-w-sm'>
                                                    {training.description}
                                                </div>

                                                <div className='mt-2'>
                                                    <div className='flex gap-x-1.5'>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === training.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>Create new datasets</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === training.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>Grow existing datasets</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div> 
                                    </Link>
                                    <Link
                                        to="/learn"
                                        className="flex items-start gap-x-2.5 p-3 font-semibold text-gray-900 rounded-md hover:bg-gray-100 h-36"
                                        onMouseEnter={() => setIsHovered(tasks.name)}
                                        >
                                            <div className='px-2 mt-1'>
                                                <tasks.icon
                                                    className={`h-6 w-6 ${(isHovered === tasks.name) ? "text-violet-600" : "text-gray-600"}`}
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div>
                                                <div>
                                                    {tasks.name}
                                                    {tasks.isBeta && (
                                                        <span className="ml-2 text-xs font-semibold bg-red-600/20 text-red-600 rounded-md px-1.5 py-0.5">
                                                            Alpha
                                                        </span>
                                                    )}
                                                </div>
                                                
                                                <div className='font-medium text-gray-500 leading-normal mt-1 max-w-sm'>
                                                    {tasks.description}
                                                </div>

                                                <div className='mt-2'>
                                                    <div className='flex gap-x-1.5'>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === tasks.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>Agent-to-agent communication</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === tasks.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>AutoGPT</span>
                                                        </div>
                                                    </div>
                                                </div>

                                            </div> 
                                    </Link>
                                    <Link
                                        to="/learn"
                                        className="flex items-start gap-x-2.5 p-3 font-semibold text-gray-900 rounded-md hover:bg-gray-100 h-36"
                                        onMouseEnter={() => setIsHovered(plugins.name)}
                                        >
                                            <div className='px-2 mt-1'>
                                                <plugins.icon
                                                    className={`h-6 w-6 ${(isHovered === plugins.name) ? "text-violet-600" : "text-gray-600"}`}
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div>
                                                <div>
                                                    {plugins.name}
                                                    {plugins.isBeta && (
                                                        <span className="ml-2 text-xs font-semibold bg-red-600/20 text-red-600 rounded-md px-1.5 py-0.5">
                                                            Alpha
                                                        </span>
                                                    )}
                                                </div>
                                                
                                                <div className='font-medium text-gray-500 leading-normal mt-1 max-w-xs'>
                                                    {plugins.description}
                                                </div>

                                                <div className='mt-2'>
                                                    <div className='flex gap-x-1.5'>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === training.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>Search</span>
                                                        </div>
                                                        <div className={`flex items-center gap-x-1.5 ${(isHovered === training.name) ? "bg-gray-200/80" : "bg-gray-100"} rounded-md px-1 py-0.5`}>
                                                            <span className='text-gray-500 text-xs'>PDF Writer</span>
                                                        </div>
                                                    </div>
                                                </div>

                                            </div> 
                                    </Link>
                                </div>
                            </div>
                        </div>
                    </Popover.Panel>
                </Transition>
            </div>
        </Popover>
    );
}



export function ResourcesDropdown(props) {

    // For whole hover
    const [isOpen, setIsOpen] = useState(false);

    // For panel hover
    const [isHovered, setIsHovered] = useState(0);

    const handleMouseEnter = () => {
        setIsOpen(true);
    };
    
    const handleMouseLeave = () => {
        setIsOpen(false);
        setIsHovered(0);
    };

    return (
        <Popover className="relative">
            <div
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                className="relative items-center text-base font-semibold text-black relative"
            >
                <span className={isOpen ? 'opacity-50' : ''}>{props.name}</span>

                <Transition
                    show={isOpen}
                    as={Fragment}
                    enter="transition ease-out duration-200"
                    enterFrom="opacity-0 translate-y-1"
                    enterTo="opacity-100 translate-y-0"
                    leave="transition ease-in duration-150"
                    leaveFrom="opacity-100 translate-y-0"
                    leaveTo="opacity-0 translate-y-1"
                >
                    <Popover.Panel className={"absolute left-1/2 z-10 pt-5 flex w-screen max-w-max -translate-x-1/2 px-4"}>
                        <div className='relative'>
                            <PopoverHump className="absolute z-50" width={22} height={14} />
                            <div className="relative z-0 flex-auto overflow-hidden rounded-md bg-white text-sm leading-6 drop-shadow-3xl ring-1 ring-gray-900/5">
                                <div className={`p-4 grid grid-cols-2`}>
                                    {props.resources.map((item) => (
                                        <Link
                                        to={item.to}
                                        className="flex items-start gap-x-2.5 p-3 font-semibold text-gray-900 rounded-md hover:bg-gray-100 h-28"
                                        onMouseEnter={() => setIsHovered(item.name)}
                                        >
                                            <div className='px-2 mt-1'>
                                                <item.icon
                                                    className={`h-6 w-6 ${(isHovered === item.name) ? "text-violet-600" : "text-gray-600"}`}
                                                    aria-hidden="true"
                                                />
                                            </div>
                                            <div>
                                                {item.name}
                                                <div className='font-medium text-gray-500 leading-snug mt-1 max-w-xs'>
                                                    {item.description}
                                                </div>
                                            </div> 
                                        </Link>
                                    ))}
                                </div>
                                <div className={`grid grid-cols-3 divide-x divide-gray-900/5 bg-gray-50`}>
                                    {props.resourcesBottom.map((item) => (
                                        <Link
                                        to={item.to}
                                        className="flex items-center justify-center gap-x-2.5 p-3 font-semibold text-gray-900 hover:bg-gray-100"
                                        onMouseEnter={() => setIsHovered(item.name)}
                                        >
                                            <item.icon className={`h-5 w-5 flex-none text-gray-400 ${(isHovered === item.name) ? "text-blue-950" : "text-gray-400"}`} aria-hidden="true"  />
                                            {item.name}

                                        </Link>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </Popover.Panel>
                </Transition>
            </div>
        </Popover>
    );
}



// No corner curve Triangle
export const Triangle = ({width, height}) => (
    <svg className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full z-50" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={`M${width/2} 0 L${width} ${height} L0 ${height} Z`} fill="white" stroke="rgba(153, 153, 153, 0.2)" strokeWidth="1" />
        <line x1="1" y1={height} x2={width - 1} y2={height} stroke="white" strokeWidth="2" />
    </svg>
);

const PopoverHump = ({width, height}) => (
    <svg className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full z-50" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <path d={`M0 ${height} Q${width/2} ${height/100} ${width} ${height} Z`} fill="white" stroke="rgba(153, 153, 153, 0.2)" strokeWidth="1" />
        <line x1="1" y1={height} x2={width - 1} y2={height} stroke="white" strokeWidth="2" />
    </svg>
);

function AgentPng({PNG, size}) {
    return (
        <img
            src={PNG}
            alt="Avatar Frame"
            className={`transition-transform duration-500 ease-in-out transform ${size === 13 ? 'scale-110' : 'scale-100'} h-14 w-auto max-w-full object-contain absolute`}
        />
    );
}
