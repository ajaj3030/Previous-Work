import { AgentsDropdown, ResourcesDropdown, InfrastructureDropdown } from './HeaderDropdown';
import LoginButton from './LogInButton';
import React from "react";
import { Link } from "react-router-dom";
import StandardAgent from "../../../assets/baby-animation/down/down-0.png";

import {
  CubeIcon,
  CpuChipIcon,
  BriefcaseIcon,
  AcademicCapIcon,
  PhoneIcon,
  RocketLaunchIcon,
  RectangleGroupIcon,
  NewspaperIcon,
  LifebuoyIcon,
} from '@heroicons/react/24/solid'

const agents = [
  { name: 'Standard', description: 'Built on GPT-4. Ready to train as you want.', href: '/pricing', icon: StandardAgent, iconType: "object" },
  
];

// UNCOMMENT WHEN ADDING DEMO TO AGENTS
const agentsBottom = [
  // { name: 'Watch demo', href: '#', icon: PlayCircleIcon },
];

const resources = [
  { name: 'Learn more', description: 'We want our customers to get the most out of their agents. Learn more about how you can.', to: '/learn', icon: AcademicCapIcon },
  { name: 'Support center', description: "Have an issue you need to troubleshoot? Our Support Agent can help you out. Yes, we use agents too!", to: '#', icon: LifebuoyIcon },
  { name: 'Customer stories', description: "Our customers use The Pin Factor to build cool things. Check out some of their stories here.", to: '#', icon: RocketLaunchIcon },
  { name: 'Contact sales', description: 'For country-specific rates and volume discounts, please contact our sales team.', to: '/sales', icon: PhoneIcon },
  { name: 'Jobs', description: "We're always looking for great people to join the team. Apply for a role here.", to: '#', icon: BriefcaseIcon },
  { name: 'Frequently asked questions', description: "Most questions we get, we've been asked before. Find your answer immediately.", to: '#', icon: RectangleGroupIcon },
];

const resourcesBottom = [
  { name: 'Become a partner', to: '/partners', icon: CpuChipIcon, },
  { name: 'Investors', to: '/investors', icon: CubeIcon },
  { name: 'Media inquiries', to: '/press', icon: NewspaperIcon },
];


export default function Header() {
    return (
        <header className="" >
            <nav className="p-4 text-black flex justify-between items-center">
              <div className='w-48 grid justify-items-start'>
                <Link to="/" className="font-bold text-lg hover:opacity-50" end>
                    The Pin Factory
                </Link>
              </div>
                <div className="flex justify-between space-x-16 items-center">
                    <div className="block rounded-lg py-2 text-sm font-semibold text-black">
                      <AgentsDropdown name="Agents" agents={agents} agentsBottom={agentsBottom}/>
                    </div>
                    {/* <div className="block rounded-lg px-4 py-2 font-semibold text-black">
                      <InfrastructureDropdown />
                    </div> */}
                    <Link to="/pricing" className="block rounded-lg py-2 font-semibold text-black hover:opacity-50">
                      Pricing
                    </Link>
                    <div className="block rounded-lg py-2 font-semibold text-black">
                      <ResourcesDropdown name="Company" resources={resources} resourcesBottom={resourcesBottom} />
                    </div>
                </div> 
              <div className='w-48 grid justify-items-end'>
                  <Link to="/signin">
                    <LoginButton />
                  </Link>
              </div>
            </nav>
        </header>
    );
  }

