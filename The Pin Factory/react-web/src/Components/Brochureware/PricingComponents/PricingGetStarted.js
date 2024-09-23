import React from 'react';
import { GetStartedButton } from '../HomeComponents/HomeLanding.js';

export default function PricingGetStarted() {
    return (
      <div className="bg-white">
        <div className="px-6 py-24 sm:px-6 sm:py-32 lg:px-8">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Create your agent today.
              <br />
              Get your first training session free.
            </h2>
            <p className="mx-auto mt-8 max-w-xl text-lg leading-7 text-gray-600">
              Legal Assisstant. Recruitment Coordinator. Speech Writer. Language Translator. Data Analyst. Social Media Manager. Customer Service Representative. Logistics Expert. Personal Assistant. Market Analyst. Academic Researcher. 
            </p>
            <p className="mx-auto mt-6 max-w-xl text-lg leading-7 text-gray-600">
              Train agents for what you need.
            </p>
            <div className="mt-10 flex justify-center item-center gap-5">
              <GetStartedButton />
              <button className="py-2 px-5 flex items-center rounded-md bg-gray-100 hover:bg-gray-50 focus:outline-none group">
                <span className="px-1 text-base font-semibold">Learn more</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    )
  }
  