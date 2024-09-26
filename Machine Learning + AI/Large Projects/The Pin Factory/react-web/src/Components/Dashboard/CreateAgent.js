import React, { useState, useEffect, useRef, useContext } from 'react';

import AvatarController from '../Brochureware/AvatarAnimation/AvatarController';
import AgentRunAcrossController from '../Brochureware/AvatarAnimation/AgentRunAcrossController';

import CreateAgentIntro from './CreateAgentForms/CreateAgentIntro';
import SelectModelPage from './CreateAgentForms/SelectModelPage';
import AllAgentsPage from './CreateAgentForms/AllAgentsPage';
import FullAgentPage from './CreateAgentForms/FullAgentPage';

import AddMemory from './CreateAgentForms/AddMemory';
import AddTraining from './CreateAgentForms/AddTraining';
import AddConnect from './CreateAgentForms/AddConnect';
import AddAgent from './CreateAgentForms/AddAgent';

import { ModelProvider, ModelContext } from './CreateAgentForms/ModelContext';

export default function CreateAgentPage() {
  return (
    <div className="relative isolate bg-white">
      <ModelProvider>
        <CreateAgentPageContent />
      </ModelProvider>
    </div>
  );
}

function CreateAgentPageContent() {

  const [currentStep, setCurrentStep] = useState(0);

  const nextStep = () => setCurrentStep(currentStep + 1);
  const prevStep = () => {
    if (currentStep == 3 && isExpanded == true) {
      setIsExpanded(false);
    }
    currentStep > 0 ? setCurrentStep(currentStep - 1) : setCurrentStep(currentStep)
  };

  let DynamicStep;

  switch (currentStep) {
    case 0:
      DynamicStep = CreateAgentIntro;
      break;
    case 1:
      DynamicStep = SelectModelPage;
      break;
    case 2:
      DynamicStep = AllAgentsPage;
      break;
    case 3:
      DynamicStep = FullAgentPage;
      break;
    default:
      DynamicStep = CreateAgentIntro;
  }

  const [isExpanded, setIsExpanded] = useState(false);

  let DynamicExpandedContent;
  const [currentExpandedContent, setCurrentExpandedContent] = useState(0);

  const toggleExpand = (selected_section) => {
    if (isExpanded) {
      if (selected_section == null) {
        setIsExpanded(!isExpanded);
        return;
      } else {
        setCurrentExpandedContent(selected_section);
        return;
      }
    } else {
      setCurrentExpandedContent(selected_section);
      setIsExpanded(!isExpanded);
      return;
    }
  };

  switch (currentExpandedContent) {
    case 0:
      DynamicExpandedContent = null;
    case 1:
      DynamicExpandedContent = AddAgent;
      break;
    case 2:
      DynamicExpandedContent = AddMemory;
      break;
    case 3:
      DynamicExpandedContent = AddTraining;
      break;
    case 4:
      DynamicExpandedContent = AddConnect;
      break;
    default:
      DynamicExpandedContent = BlankSlider;
  }

  const ref = useRef(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);

  useEffect(() => {
    setWidth(ref.current.offsetWidth);
    setHeight(ref.current.offsetHeight);
  }, []);

  const { state, setState } = useContext(ModelContext);

  const [agentsRunning, setAgentsRunning] = useState(true);
  useEffect(() => {
    console.log("changeAgentsRunning()")
    if (currentStep > 1) {
      setAgentsRunning(false);
    }
    return;
  }, [currentStep]);

  
  return (
      <div className="grid grid-cols-2">

      {/* LEFT SIDE OF CONTENT */}
      <div className='relative'>
        {/* UnderneathSlider */}
        <div className={`absolute top-0 left-0 z-30 w-full overflow-hidden transition-transform ease-in-out ${isExpanded ? 'translate-x-full duration-700' : 'duration-300'}`}>
          <div className='flex justify-start h-screen bg-gray-50 border-1 border-r'>
            <DynamicExpandedContent toggleExpand={toggleExpand} />
          </div>
        </div>
        {/* Actual Left side content */}
        <div className={`${isExpanded ? "fixed" : "absolute"} top-0 left-0 z-40 ${isExpanded ? "w-1/2" : "w-full"} overflow-hidden`}>
          <div className={`flex justify-start min-h-screen h-full bg-white border-1 border-l border-r`}>
            <DynamicStep onNext={nextStep} onPrev={prevStep} toggleExpand={toggleExpand} />
          </div>
        </div>
      </div>

      {/* RIGHT SIDE OF CONTENT */}
      <div className='relative h-screen'>

        {/* BACKGROUND */}
          <div className="fixed top-0 right-0 z-10 h-screen overflow-hidden bg-gray-100 ring-1 ring-gray-900/10 w-1/2">
              <svg
                  className="absolute inset-0 h-full w-full stroke-blue-900/30 [mask-image:radial-gradient(100%_100%_at_top_left,#FFFFFFFF,#FFFFFF8C)]"
                  aria-hidden="true"
              >
              <defs>
                <pattern
                  id="83fd4e5a-9d52-42fc-97b6-718e5d7ee527"
                  width={200}
                  height={200}
                  x="100%"
                  y={-1}
                  patternUnits="userSpaceOnUse"
                >
                  <path d="M130 200V.5M.5 .5H200" stroke-dasharray="6,6" stroke-width="1.2" fill="none" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" strokeWidth={0} fill="#f9fafb" />
              <svg x="100%" y={-1} className="overflow-visible fill-gray-100">
                <path d="M-470.5 0h201v201h-201Z" strokeWidth={0} />
              </svg>
              <svg x="100%" y={-1} className="overflow-visible fill-gray-100">
                <path d="M-470.5 0h201v201h-201Z" strokeWidth={0} />
              </svg>
              <rect width="100%" height="100%" strokeWidth={0} fill="url(#83fd4e5a-9d52-42fc-97b6-718e5d7ee527)" />
            </svg>
          </div>

          {/* AVATAR ANIMATION ON TOP OF BACKGROUND */}
          <div ref={ref} className='fixed top-0 right-0 z-20 w-1/2 h-screen'>
            <AgentRunAcrossController agentsRunning={agentsRunning} />
            {!agentsRunning && 
              <AvatarController containerWidth={width} containerHeight={height} agents_input={state.agents} />
            }
          </div>
      </div>
    </div>
  )
}

function BlankSlider() {
  return (
    <div className='flex justify-center h-screen bg-gray-50'>
      <p>DEFAULT</p>
    </div>
  );
};