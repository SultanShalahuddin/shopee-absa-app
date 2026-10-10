import { AspectPrediction, Sentiment } from "@/lib/api";

const config: Array<{ key: Sentiment; label: string; color: string }> = [
  { key: "positif", label: "Positif", color: "#38c997" },
  { key: "negatif", label: "Negatif", color: "#ff7d68" },
  { key: "netral", label: "Netral", color: "#8ea0b9" },
  { key: "tidak_disebut", label: "Tidak disebut", color: "#dbe3ec" },
];

export default function SentimentSummary({ predictions }: { predictions: AspectPrediction[] }) {
  const counts = Object.fromEntries(config.map(({ key }) => [key, predictions.filter((x) => x.sentiment === key).length])) as Record<Sentiment, number>;
  const total = Math.max(predictions.length, 1);
  let cursor = 0;
  const stops = config.map(({ key, color }) => {
    const start = cursor;
    cursor += (counts[key] / total) * 100;
    return `${color} ${start}% ${cursor}%`;
  }).join(", ");

  return (
    <aside className="summary-card">
      <div>
        <span className="eyebrow">Ringkasan</span>
        <h2>Sentimen review</h2>
      </div>
      <div className="donut" style={{ background: `conic-gradient(${stops})` }}>
        <div><strong>{total}</strong><span>aspek</span></div>
      </div>
      <div className="legend">
        {config.map((item) => (
          <div key={item.key}>
            <span className="legend-dot" style={{ background: item.color }} />
            <span>{item.label}</span>
            <strong>{counts[item.key]}</strong>
          </div>
        ))}
      </div>
    </aside>
  );
}
