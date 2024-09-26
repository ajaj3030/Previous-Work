import React, { useState } from 'react';
import { ArrowRight, BookOpen, Loader2 } from 'lucide-react';

const App = () => {
  const [field, setField] = useState('');
  const [techLevel, setTechLevel] = useState('intermediate');
  const [summaryLength, setSummaryLength] = useState('medium');
  const [papers, setPapers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPapers([]); // Clear previous results
    try {
      const response = await fetch('http://localhost:8001/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ field, techLevel, summaryLength }),
      });
      if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
      const data = await response.json();
      setPapers(data.papers);
    } catch (error) {
      console.error('Error fetching papers:', error);
      setError('Failed to fetch papers. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-600 to-indigo-900 text-white flex flex-col">
      <header className="bg-indigo-700 shadow-lg">
        <div className="container mx-auto py-6 px-4 flex items-center justify-between">
          <h1 className="text-4xl font-extrabold">SciDaily Paper Summarizer</h1>
          <span className="text-indigo-200 font-semibold">Powered by SciDaily</span>
        </div>
      </header>
      <main className="flex-grow">
        <div className="container mx-auto px-4 py-12">
          <form onSubmit={handleSubmit} className="space-y-8">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-8">
              <div>
                <label htmlFor="field" className="block text-lg font-semibold">
                  Research Field
                </label>
                <input
                  type="text"
                  name="field"
                  id="field"
                  className="mt-2 block w-full px-4 py-3 bg-indigo-800 text-white border border-indigo-600 rounded-lg shadow-sm focus:ring-indigo-300 focus:border-indigo-300"
                  placeholder="e.g., quantum computing"
                  value={field}
                  onChange={(e) => setField(e.target.value)}
                  required
                />
              </div>
              <div>
                <label htmlFor="techLevel" className="block text-lg font-semibold">
                  Technical Level
                </label>
                <select
                  id="techLevel"
                  name="techLevel"
                  className="mt-2 block w-full px-4 py-3 bg-indigo-800 text-white border border-indigo-600 rounded-lg shadow-sm focus:ring-indigo-300 focus:border-indigo-300"
                  value={techLevel}
                  onChange={(e) => setTechLevel(e.target.value)}
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="expert">Expert</option>
                </select>
              </div>
            </div>
            <div>
              <label htmlFor="summaryLength" className="block text-lg font-semibold">
                Summary Length
              </label>
              <select
                id="summaryLength"
                name="summaryLength"
                className="mt-2 block w-full px-4 py-3 bg-indigo-800 text-white border border-indigo-600 rounded-lg shadow-sm focus:ring-indigo-300 focus:border-indigo-300"
                value={summaryLength}
                onChange={(e) => setSummaryLength(e.target.value)}
              >
                <option value="short">Short</option>
                <option value="medium">Medium</option>
                <option value="long">Long</option>
              </select>
            </div>
            <div>
              <button
                type="submit"
                className="w-full flex justify-center py-4 px-6 border border-transparent rounded-lg shadow-xl text-lg font-semibold text-white bg-indigo-500 hover:bg-indigo-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-400"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" />
                    Summarizing...
                  </>
                ) : (
                  <>
                    Summarize Papers
                    <ArrowRight className="ml-2 -mr-1 h-5 w-5" aria-hidden="true" />
                  </>
                )}
              </button>
            </div>
          </form>

          {error && (
            <div className="mt-6 bg-red-600 border-l-4 border-red-800 p-4 rounded-lg">
              <div className="flex">
                <div className="flex-shrink-0">
                  <svg
                    className="h-5 w-5 text-white"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <path
                      fillRule="evenodd"
                      d="M18 8A10 10 0 11.867 12.501a.5.5 0 01-.48-.36.5.5 0 01.359-.641A9 9 0 109 18a.5.5 0 01.5-.5.5.5 0 01.5.5 10 10 0 109-10zM8 9v3a1 1 0 102 0V9a1 1 0 10-2 0zM9 13a1 1 0 110 2 1 1 0 010-2z"
                      clipRule="evenodd"
                    />
                  </svg>
                </div>
                <div className="ml-3">
                  <h3 className="text-lg font-semibold text-white">Error</h3>
                  <div className="mt-2 text-sm text-white">
                    <p>{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="mt-12 space-y-8">
            {papers.map((paper, index) => (
              <div
                key={index}
                className="bg-indigo-800 shadow-xl overflow-hidden sm:rounded-lg transition duration-300 ease-in-out transform hover:scale-105 hover:shadow-2xl"
              >
                <div className="px-6 py-5 sm:px-8 bg-indigo-700">
                  <h3 className="text-2xl leading-6 font-extrabold text-white">{paper.title}</h3>
                  <p className="mt-1 text-sm text-indigo-200">
                    Published: {new Date(paper.published).toLocaleString()}
                  </p>
                </div>
                <div className="border-t border-indigo-600 px-6 py-5 sm:p-8">
                  <dl>
                    <div className="py-4">
                      <dt className="text-sm font-semibold text-indigo-200">Summary</dt>
                      <dd className="mt-1 text-sm text-white">{paper.summary}</dd>
                    </div>
                    <div className="py-4">
                      <dt className="text-sm font-semibold text-indigo-200">Full Paper</dt>
                      <dd className="mt-1 text-sm">
                        <a
                          href={paper.link}
                          className="text-indigo-300 hover:text-indigo-100 flex items-center group"
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Read full paper
                          <BookOpen
                            className="ml-2 h-5 w-5 transition-transform duration-150 ease-in-out group-hover:translate-x-1"
                            aria-hidden="true"
                          />
                        </a>
                      </dd>
                    </div>
                  </dl>
                </div>
              </div>
            ))}
          </div>
        </div>
      </main>
      <footer className="bg-indigo-700">
        <div className="container mx-auto py-4 px-4 text-center">
          <p className="text-sm text-indigo-200">
            &copy; {new Date().getFullYear()} SciDaily. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default App;
