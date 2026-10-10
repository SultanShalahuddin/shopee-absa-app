import { AspectPrediction } from "@/lib/api";
import { BadgeDollarSign, BadgePercent, Box, CreditCard, KeyRound, Megaphone, RefreshCcw, Smartphone } from "lucide-react";

const icons = {
  iklan_dan_konten: Megaphone,
  pengiriman_dan_kurir: Box,
  performa_aplikasi: Smartphone,
  akun_login_keamanan: KeyRound,
  pembayaran_cod_dan_keuangan: CreditCard,
  promo_voucher_harga_dan_biaya: BadgePercent,
  penjual_dan_kualitas_produk: BadgeDollarSign,
  retur_refund_dan_layanan: RefreshCcw,
};

const labels = {
  tidak_disebut: "Tidak disebut",
  negatif: "Negatif",
  netral: "Netral",
  positif: "Positif",
};

export default function SentimentCard({ prediction }: { prediction: AspectPrediction }) {
  const Icon = icons[prediction.aspect as keyof typeof icons] || Smartphone;
  const percentage = Math.round(prediction.confidence * 100);

  return (
    <article className={`aspect-card sentiment-${prediction.sentiment}`}>
      <div className="aspect-heading">
        <span className="aspect-icon"><Icon size={19} /></span>
        <span className={`sentiment-pill ${prediction.sentiment}`}>{labels[prediction.sentiment]}</span>
      </div>
      <h3>{prediction.aspect_label}</h3>
      <div className="confidence-label">
        <span>Tingkat keyakinan</span><strong>{percentage}%</strong>
      </div>
      <div className="confidence-track" aria-label={`Confidence ${percentage}%`}>
        <span style={{ width: `${percentage}%` }} />
      </div>
    </article>
  );
}
