import React, {useState} from 'react';
import { ArrowStalk, ArrowHead } from '../Brochureware/Header/LogInButton.js';
import { LockClosedIcon, LockOpenIcon } from '@heroicons/react/20/solid';

export default function PlugInFront({plugin, index}) {
    const [buttonHovered, setButtonHovered] = useState(false);
    const [childHovered, setChildHovered] = useState(false);
    const handleHoverChange = (isHovered) => {
        setChildHovered(isHovered);
    };
    return (

        <button
        onMouseEnter={() => setButtonHovered(true)}
        onMouseLeave={() => setButtonHovered(false)}
        className={`flex flex-row w-full items-start h-48 justify-between rounded-md border py-3 focus:outline-none group px-4 bg-white shadow-[0_0px_8px_-1px_rgba(0,0,0,0.1)] border-gray-900/20`}
        >
            
            <div className="flex flex-row w-full h-full justify-start items-start gap-x-4 items-start mt-1">

                {/* LEFT IMAGE */}
                <div className='flex flex-row items-center justify-center bg-black w-12 aspect-square rounded-md outline outline-2 outline-gray-900/10 overflow-hidden'>
                    <img
                        src={
                        plugin.logo
                        }
                        className={`w-6 aspect-square`}
                    />
                </div>

                {/* MAIN CONTENT */}
                <div className={`flex flex-col justify-between w-full h-full text-start pb-3 pt-1 space-y-3`}>

                    {/* TOP 2 OF MAIN CONTENT */}
                    <div className={`flex flex-col space-y-1 text-start`}>
                        {/* TOP LINE */}
                        <div className={`flex items-center space-x-4 mr-2 ${buttonHovered ? 'opacity-50' : ''}`}>
                            {/* NAME */}
                            <div className='flex items-center'>
                                <p className={`font-semibold leading-6 text-gray-900 line-clamp-1`}>{plugin.name}</p>
                            </div>
                        </div>
                        {/* MAIN TEXT BODY */}
                        <p className={`text-sm leading-4.5 text-gray-500/90 line-clamp-3`}>{plugin.description}</p>
                        
                    </div>

                    {/* BOTTOM OF MAIN */}
                    <div className="flex flex-row items-center justify-start justify-items-start w-full">
                        {/* IF PLUGIN IS CONNECTED, ORANGE CONNECTED BUTTON ELSE UNCONNECTED BUTTON */}
                        {plugin.state === "CONNECTED" ? (
                            <button className={`flex flex-row items-center justify-center px-3 py-1.5 rounded-md bg-orange-600/90 text-white text-sm font-semibold`}>
                                <LockClosedIcon className={`w-4 h-4 mr-1`}/>
                                <p>Connected</p>
                            </button>
                        ) : (
                            <button className={`flex flex-row items-center justify-center px-3 py-1.5 rounded-md bg-white text-gray-900 text-sm font-semibold border border-gray-900/20`}>
                                <LockOpenIcon className={`w-4 h-4 mr-1`}/>
                                <p>Connect</p>
                            </button>
                        )}
                    </div>

                </div>
            </div>
        </button>
    );
}