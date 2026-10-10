import { HistoryItem } from "@/lib/api";
import { Clock3, Database } from "lucide-react";

export default function HistoryList({ items }: { items: HistoryItem[] }) {
  return (
    <section className="history-section" id="riwayat">
      <div className="section-title-row">
        <div>
          <span className="eyebrow">Database</span>
          <h2>Riwayat analisis</h2>
          <p>Review yang tersimpan di Supabase.</p>
        </div>
        <span className="history-count"><Database size={17} /> {items.length} data terbaru</span>
      </div>
      {items.length === 0 ? (
        <div className="empty-history">Belum ada riwayat atau Supabase belum dikonfigurasi.</div>
      ) : (
        <div className="history-list">
          {items.map((item) => {
            const mentioned = item.predictions.filter((p) => p.sentiment !== "tidak_disebut");
            return (
              <article className="history-item" key={item.id}>
                <p>{item.review_text}</p>
                <div className="history-meta">
                  <span><Clock3 size={14} /> {new Date(item.created_at).toLocaleString("id-ID")}</span>
                  <span>{item.latency_ms} ms</span>
                  <span>{item.model_version}</span>
                </div>
                <div className="history-badges">
                  {mentioned.slice(0, 4).map((p) => <span className={p.sentiment} key={p.aspect}>{p.aspect_label}: {p.sentiment}</span>)}
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
