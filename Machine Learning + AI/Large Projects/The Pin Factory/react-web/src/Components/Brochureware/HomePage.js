import React from "react";
import HomeLanding from "./HomeComponents/HomeLanding.js";
import AdditionalContent from "./HomeComponents/HomePage2.js";
import LearnMore_Agents from "./HomeComponents/LearnMore_Agents.js";
import "./HomeComponents/HomePage.css";
import Header from "./Header/Header.js";
import Footer from "./Footer/Footer.js";

export const brochurewareMargin = 192;

export default function HomePage() {
	return (
		<div className="relative z-0 backdrop">
			<div className="absolute top-0 left-0 right-0 z-10 mx-44 px-6 py-2">
				<Header />
			</div>
			<div className="fixed top-0 left-0 right-0 bottom-0 z-0">
				<HomeLanding />
				<div className="relative z-1">
					{/* <div className="bg-gray-100 relative z-10">
            <LearnMore_Agents className="relative z-20" />
          </div>
          <Footer />
          <div className="fixed top-0 left-0 bottom-0 z-15 border-l border-dashed border-gray-300 mx-44" />
          <div className="fixed top-0 right-0 bottom-0 z-15 border-r border-dashed border-gray-300 mx-44" /> */}
				</div>
			</div>
		</div>
	);
}
