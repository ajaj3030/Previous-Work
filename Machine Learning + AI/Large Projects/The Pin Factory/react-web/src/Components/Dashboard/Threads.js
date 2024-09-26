import React from "react";
import DashboardHeader from "./DashboardHeader.js";
import ChatView from "./ChatView.js";
import { ReactFlowProvider } from "reactflow";
import "reactflow/dist/style.css";

export default function DashboardHome() {
	const HeaderHeight = 8; // in 'rem'

	return (
		<ReactFlowProvider>
			<div className="relative h-screen bg-transparent">
				{/* HEADER */}
				{/* <DashboardHeader HeaderHeight={HeaderHeight} /> */}
				{/* CONTENT */}
				<div
					className="relative z-0 flex h-full bg-transparent backdrop"
					style={{ paddingTop: `${0}vh` }}>
					<ChatView />
				</div>
			</div>
		</ReactFlowProvider>
	);
}
