import React from 'react';
import './HomeComponents/HomePage.css';
import Header from './Header/Header.js';
import Footer from './Footer/Footer.js';
import { brochurewareMargin } from './HomePage.js';

import PricingTop from './PricingComponents/PricingTop.js';
import PricingTestimonials from './PricingComponents/PricingTestimonials.js';
import PricingGetStarted from './PricingComponents/PricingGetStarted.js';

export default function PricingPage() {
  return (
    <div className='relative z-0' style={{ marginLeft: `${brochurewareMargin}px`, marginRight: `${brochurewareMargin}px`  }}>
      <div className='absolute top-0 left-0 right-0 z-10'>
        <Header />
      </div>
      <div className='absolute top-0 left-0 right-0 bottom-0 z-5 home-page'>
        <PricingTop />
        <PricingTestimonials />
        <PricingGetStarted />
        <Footer />
      </div>
    </div>
  );
};