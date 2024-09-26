import React, { useEffect, useRef, useState, useContext } from "react";
import DashboardHeader from "./DashboardHeader.js";
import { Search, XMark, PencilIcon, ThreadsIcon } from './SVG_icons.js'
import PlugInFront from "./PlugInFront.js";
import SquareLogo from "../../assets/square-logos/Logo_package_USEN/PNG_RGB/Square_Jewel_White_cropped.png";


var pluginsToDisplay = [
    {
        id: "1",
        name: "Square",
        description: "Square, Inc. is an American financial services and digital payments company based in San Francisco, California. The company markets software and hardware payments products and has expanded into small business services.",
        logo: SquareLogo, // Please replace this URL with the real logo URL.
        url: "https://squareup.com/",
        state: "NOT CONNECTED" // for instance, it could be active, inactive, deprecated
    },
    {
        id: "1",
        name: "Square",
        description: "Square, Inc. is an American financial services and digital payments company based in San Francisco, California. The company markets software and hardware payments products and has expanded into small business services.",
        logo: SquareLogo, // Please replace this URL with the real logo URL.
        url: "https://squareup.com/",
        state: "CONNECTED" // for instance, it could be active, inactive, deprecated
    },
];




export default function PlugIns() {
    
    const HeaderHeight = 8;
    const [searchValue, setSearchValue] = useState("");
	const [newThread, setNewThread] = useState(false);
	const [selectedNavButton, setSelectedNavButton] = useState("Connected");

    const connectedRef = useRef(null);
	const allRef = useRef(null);
	const merchantRef = useRef(null);

	const [widths, setWidths] = useState({'Connected': 0, 'All': 0, 'Merchant': 0});
	const [heights, setHeights] = useState({'Connected': 0, 'All': 0, 'Merchant': 0});

	useEffect(() => {
		setWidths({'Connected': connectedRef.current.offsetWidth, 'All': allRef.current.offsetWidth, 'Merchant': merchantRef.current.offsetWidth});
		setHeights({'Connected': connectedRef.current.offsetHeight, 'All': allRef.current.offsetHeight, 'Merchant': merchantRef.current.offsetHeight});
	}, []);

	const [selectedNavUnderlineBorderLeftMargin, setSelectedNavUnderlineBorderLeftMargin] = useState(0);

	useEffect(() => {
		// Update the selectedNavUnderlineBorderLeftMargin to be the sum of the widths of the nav buttons to the left of the selected one plus the spacing between them
		let sum = 20;
		for (let i = 0; i < Object.keys(widths).length; i++) {
			if (Object.keys(widths)[i] === selectedNavButton) {
				break;
			}
			sum += widths[Object.keys(widths)[i]];
			sum += 32;
		}
		setSelectedNavUnderlineBorderLeftMargin(sum + (Object.keys(widths).length - 1) * 4);
	}, [selectedNavButton]);
    return (
		<div className="relative h-screen bg-transparent">
            {/* HEADER */}
            <DashboardHeader HeaderHeight={HeaderHeight} />
            {/* CONTENT */}
            <div className="relative z-0 flex h-full bg-transparent" style={{ paddingTop: `${HeaderHeight}vh` }}>
                
                <div className="relative w-full h-full bg-white">

                    {/* THREADS HEADER */}
                    <h1 className="text-xl font-semibold text-gray-900 px-8 pt-6">Plug Ins</h1>
                    <h1 className="px-8 text-sm font-medium text-gray-500 pt-1">Connect agents to the real world.</h1>
                    <div className="relative px-8 py-0 border-b border-gray-200 w-full pt-2">
                        <div className="relative flex flex-row space-x-2 w-full justify-between items-center">
                            {/* NAV BAR - LEFT SIDE UNDER HEADER */}
                            <div className="flex flex-row items-center justify-items-start justify-start space-x-8">
                                <button
                                    ref={connectedRef}
                                    onClick={() => setSelectedNavButton("Connected")}
                                    className={`text-sm font-semibold ${selectedNavButton === 'Connected' ? 'text-orange-600/80' : 'text-gray-900'}`}>
                                    Connected
                                </button>
                                <button 
                                    ref={allRef}
                                    onClick={() => setSelectedNavButton("All")}
                                    className={`text-sm font-semibold ${selectedNavButton === 'All' ? 'text-orange-600/80' : 'text-gray-900'}`}>
                                    All
                                </button>
                                <button 
                                    ref={merchantRef}
                                    onClick={() => setSelectedNavButton("Merchant")}
                                    className={`text-sm font-semibold ${selectedNavButton === 'Merchant' ? 'text-orange-600/80' : 'text-gray-900'}`}>
                                    Merchant
                                </button>
                            </div>
                            {/* RIGHT SIDE */}
                            <div className="flex flex-row items-center justify-start space-x-4 pb-2">
                                {/* SEARCH BAR */}
                                <div className="flex flex-row items-center justify-between w-96 rounded-full bg-gray-200/70 px-4 py-2">
                                    <div className="flex flex-row items-center justify-start space-x-2 w-full">
                                        {/* Search Icon */}
                                        <Search
                                            size={"5"}
                                            color={"#4b5563"}
                                            strokeWidth={"0.4"}
                                        />
                                        {/* Search Input */}
                                        <input
                                            type="text"
                                            value={searchValue}
                                            placeholder="Search plug-ins"
                                            onChange={(e) => setSearchValue(e.target.value)} // Step 3
                                            className="text-sm text-gray-900 bg-transparent outline-none w-full"
                                        />
                                    </div>
                                    {/* Empty Search Button */}
                                    {searchValue && (
                                        <button onClick={() => setSearchValue("")}>
                                            <XMark
                                                size={"5"}
                                                color={"#4b5563"}
                                                strokeWidth={"0.4"}
                                            />
                                        </button>
                                    )}
                                </div>
                            </div>
                        </div>
                        {/* Border than underlines nav bar - left margin calculated by selected item and width */}
                        <div 
                            className="absolute bottom-0 left-0 z-10 h-[3px] translate-y-1/2 bg-orange-600/80 transition-all duration-300 ease-in-out" 
                            style={{ width: `${(widths[selectedNavButton]+12)}px`, transform: `translateX(${-5}px)`, marginLeft: `${selectedNavUnderlineBorderLeftMargin}px`}}>
                        </div>
                    </div>
                    {/* END THREADS HEADER */}

                    <div className="px-8 py-4">
                        {pluginsToDisplay.length === 0 ? (
                            <div className="flex w-full justify-center text-sm font-medium text-gray-400 mt-24">
                                We couldn't find any plug-ins matching your search.
                            </div>
                        ) : (
                            <div className="grid grid-cols-3 gap-4">
                                {pluginsToDisplay.map((plugin, index) => (
                                    <PlugInFront
                                        plugin={plugin}
                                        index={index}
                                    />
                                ))}
                            </div>
                        )}
                    </div>

                </div>
            </div>
        </div>
	);
}