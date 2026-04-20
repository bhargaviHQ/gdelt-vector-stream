import { useState } from "react";
import Header from "./components/Header";
import SearchTab from "./components/SearchTab";
import AskTab from "./components/AskTab";

type Tab = "search" | "ask";

function App() {
  const [activeTab, setActiveTab] = useState<Tab>("search");

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Header />

      <main className="mx-auto max-w-4xl px-6 py-6 space-y-6">
        {/* Tab toggle */}
        <div className="flex gap-1 rounded-lg bg-gray-800/50 p-1 w-fit">
          <button
            onClick={() => setActiveTab("search")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              activeTab === "search"
                ? "bg-gray-700 text-white"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Search
          </button>
          <button
            onClick={() => setActiveTab("ask")}
            className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
              activeTab === "ask"
                ? "bg-gray-700 text-white"
                : "text-gray-400 hover:text-gray-200"
            }`}
          >
            Ask
          </button>
        </div>

        {activeTab === "search" ? <SearchTab /> : <AskTab />}
      </main>
    </div>
  );
}

export default App;
