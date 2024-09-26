import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import Threads from "./Components/Dashboard/Threads.js";
import PlugIns from "./Components/Dashboard/PlugIns.js";
import Agents from "./Components/Dashboard/Agents.js";

import HomePage from "./Components/Brochureware/HomePage.js";
import ContactSalesPage from "./Components/Brochureware/ContactSalesPage.js";
import InvestorsPage from "./Components/Brochureware/InvestorsPage.js";
import PartnersPage from "./Components/Brochureware/PartnersPage.js";
import PressInquiriesPage from "./Components/Brochureware/PressInquiriesPage.js";
import PricingPage from "./Components/Brochureware/PricingPage.js";
import SignIn from "./Components/Account/SignIn.js";
import CreateAccount from "./Components/Account/CreateAccount.js";
import AgentPage from "./Components/Dashboard/AgentsComponents/AgentPage.js";
import CreateAgentPage from "./Components/Dashboard/CreateAgent.js";
import { TaskProvider } from "./Components/Dashboard/TaskContext";
import Settings from "./Components/Dashboard/Settings";
import PrivateRoute from "./PrivateRoute.js";
import Accounts from "./Components/Dashboard/Accounts";
import Overview from "./Components/Dashboard/Overview";


function App() {
	return (
		<TaskProvider>
			<Router>
				<Routes>
					<Route path="/" element={<HomePage />} />

					<Route path="/pricing" element={<PricingPage />} />
					<Route path="/learn" element={<HomePage />} />
					<Route path="/about" element={<HomePage />} />

					<Route path="/sales" element={<ContactSalesPage />} />
					<Route path="/investors" element={<InvestorsPage />} />
					<Route path="/press" element={<PressInquiriesPage />} />
					<Route path="/partners" element={<PartnersPage />} />

					<Route path="/signin" element={<SignIn />} />
					<Route path="/createaccount" element={<CreateAccount />} />

					<Route
						path="/plugins"
						element={
							<PrivateRoute>
								<PlugIns />
							</PrivateRoute>
						}
					/>

					<Route
						path="/threads"
						element={
							<PrivateRoute>
								<Threads />
							</PrivateRoute>
						}
					/>

					<Route
						path="/agents"
						element={
							<PrivateRoute>
								<Agents />
							</PrivateRoute>
						}
					/>

					<Route
						path="/createagent"
						element={
							<PrivateRoute>
								<CreateAgentPage />
							</PrivateRoute>
						}
					/>

					<Route
						path="/settings"
						element={
							<PrivateRoute>
								<Settings />
							</PrivateRoute>
						}
					/>
					<Route
						path="/accounts"
						element={
							<PrivateRoute>
								<Accounts />
							</PrivateRoute>
						}
					/>
					<Route
						path="/overview"
						element={
							<PrivateRoute>
								<Overview />
							</PrivateRoute>
						}
					/>
				</Routes>
			</Router>
		</TaskProvider>
	);
}

export default App;
