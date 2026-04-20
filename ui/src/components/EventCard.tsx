import type { SearchResult } from "../types";

function scoreColor(score: number): string {
  if (score > 0.8) return "bg-green-900/50 text-green-300";
  if (score > 0.6) return "bg-yellow-900/50 text-yellow-300";
  return "bg-gray-700/50 text-gray-300";
}

function toneColor(tone: number): string {
  if (tone > 0) return "text-green-400";
  if (tone < 0) return "text-red-400";
  return "text-gray-400";
}

export default function EventCard({ result }: { result: SearchResult }) {
  const { metadata: m, similarity_score: score } = result;

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800/50 p-4 space-y-2">
      {/* Top row */}
      <div className="flex items-center gap-2 flex-wrap">
        <span
          className={`text-xs font-mono px-2 py-0.5 rounded ${scoreColor(score)}`}
        >
          {(score * 100).toFixed(1)}%
        </span>
        <span className="text-sm text-gray-400">{m.event_date}</span>
        <span className="text-xs bg-gray-700 text-gray-300 px-2 py-0.5 rounded">
          {m.country_code}
        </span>
        <span className="text-xs font-mono bg-gray-700/50 text-gray-400 px-2 py-0.5 rounded">
          {m.event_code}
        </span>
      </div>

      {/* Actors */}
      <p className="text-sm text-gray-100">
        {m.actor1_name || "Unknown"}{" "}
        <span className="text-gray-500">&rarr;</span>{" "}
        {m.actor2_name || "Unknown"}
      </p>

      {/* Bottom row */}
      <div className="flex items-center gap-4 text-xs">
        <span className={toneColor(m.avg_tone)}>
          Tone: {m.avg_tone.toFixed(1)}
        </span>
        <span className="text-gray-400">
          {m.num_mentions} mention{m.num_mentions !== 1 ? "s" : ""}
        </span>
        {m.source_url && (
          <a
            href={m.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 truncate max-w-xs"
          >
            Source &nearr;
          </a>
        )}
      </div>
    </div>
  );
}
