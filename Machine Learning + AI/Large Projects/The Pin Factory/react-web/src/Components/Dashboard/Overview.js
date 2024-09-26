import React, { useEffect, useRef, useState, useContext } from "react";
import DashboardHeader from "./DashboardHeader";
import AvatarController from "../Brochureware/AvatarAnimation/AvatarController";
const { v4: uuidv4 } = require("uuid");


const test_agents = [
	{
		id: uuidv4(),
		name: "Colin",
		model: {
			id: "gpt-3",
			name: "GPT-3",
			status: "Recommended",
			description:
				"GPT-3 is a language model that uses deep learning to produce human-like text. It takes in a prompt, and attempts to complete it.",
			capabilities: {
				memory: true,
				training: true,
				communication: true,
				plugins: true,
			},
		},
		avatar: {
			uid: uuidv4(),
			id: "BASIC",
			sex: {
				id: "male",
				pronouns: {
					pronoun: "he",
					objective: "him",
					possessive: "his",
				},
			},
		},
		setup: {
			agent: "COMPLETED",
			memory: "COMPLETED",
			training: "COMPLETED",
			connect: "COMPLETED",
		},
	},
];


export default function Overview() {
	const HeaderHeight = 8;
	const ref = useRef(null);
	const [width, setWidth] = useState(0);
	const [height, setHeight] = useState(0);
	useEffect(() => {
		setWidth(ref.current.offsetWidth);
		setHeight(ref.current.offsetHeight);
	}, []);
	return (
		<div className="relative h-screen bg-transparent">
			{/* HEADER */}
			{/* <DashboardHeader HeaderHeight={HeaderHeight} /> */}
			{/* CONTENT */}
			<div className="relative z-0 flex h-full w-full bg-transparent" style={{ paddingTop: `${HeaderHeight}vh` }}>
				<div className="relative h-full w-full shadow-[inset_0_0px_50px_20px_#9ca3af]">
					<div className="relative h-full absolute top-0 left-0 right-0">
						<div
						className={`relative h-full`}
						>
							{/* BACKGROUND */}
							<div className="w-full h-full ring-1 ring-gray-900/10">
								<svg
								className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_top_right,#FFFFFFFF,#FFFFFF8C)]"
								aria-hidden="true">
								<defs>
									<pattern
									id="83fd4e5a-9d52-42fc-97b6-718e5d7ee527"
									width={200}
									height={200}
									x="100%"
									y={-1}
									patternUnits="userSpaceOnUse">
									<path
										d="M130 200V.5M.5 .5H200"
										strokeDasharray="6,6"
										strokeWidth="1.2"
										fill="none"
									/>
									</pattern>
								</defs>
								<rect
									width="100%"
									height="100%"
									strokeWidth={0}
									fill="#f9fafb"
								/>
								<svg
									x="100%"
									y={-1}
									className="overflow-visible fill-gray-100">
									<path
									d="M-470.5 0h201v201h-201Z"
									strokeWidth={0}
									/>
								</svg>
								<svg
									x="100%"
									y={-1}
									className="overflow-visible fill-gray-100">
									<path
									d="M-470.5 0h201v201h-201Z"
									strokeWidth={0}
									/>
								</svg>
								<rect
									width="100%"
									height="100%"
									strokeWidth={0}
									fill="url(#83fd4e5a-9d52-42fc-97b6-718e5d7ee527)"
								/>
								</svg>
							</div>

							{/* Avatar Controller*/}
							<div ref={ref} className="absolute left-0 top-0 right-0 z-20 h-full">
								<div className="px-6 lg:px-8 mx-auto max-w-xl">
								<AvatarController containerWidth={width} containerHeight={height} agents_input={test_agents} />
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
}
