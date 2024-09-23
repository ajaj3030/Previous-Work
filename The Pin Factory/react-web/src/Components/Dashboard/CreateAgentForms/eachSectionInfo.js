// Desc: Information for each step in the create agent form
import { Square3Stack3DIcon, CodeBracketSquareIcon, AcademicCapIcon, FingerPrintIcon } from '@heroicons/react/20/solid'

export const allSections = [

    {
        index: 1,
        name: 'agent',
        icon: FingerPrintIcon,
        description: 'Finalise your agent with a role and a description.',
        bullets: [
            'Choose a role for your agent',
            'Give your agent a name',
            'Give your agent a description'
        ]
    },

    {
        index: 2,
        name: 'memory',
        icon: Square3Stack3DIcon,
        description:'Information you upload here will be remembered by your agent.',
        bullets: [
            'Memory is factual information your agent will remember across conversations.',
            'The information your agent remembers can then used to answer questions and complete tasks.',
            "It stops agents from hallucinating facts and helps them to remember information specific to you or your company.",
            'Examples might include: financial reports, product catalogues, terms of service agreements, legal documents, or just a list of your favourite restaurants.',
        ]
    },

    {
        index: 3,
        name: 'training',
        icon: AcademicCapIcon,
        description: 'Operations Expert. Lead Generation. Customer Service. Specialise your agent for specific jobs here.',
        bullets: [
            'Choose a role for your agent',
            'Give your agent a name',
            'Give your agent a description'
        ]
    },

    {
        index: 4,
        name: 'connect',
        icon: CodeBracketSquareIcon,
        description: 'Connect your agent to Plug-Ins so they can action tasks in the real world.',
        bullets: [
            'Choose a role for your agent',
            'Give your agent a name',
            'Give your agent a description'
        ]
    },

]

export function getSectionDescription(name) {
    const section = allSections.find(section => section.name === name);
    return section ? section.description : null;
}

export function getSectionBullets(name) {
    const section = allSections.find(section => section.name === name);
    return section ? section.bullets : null;
}
