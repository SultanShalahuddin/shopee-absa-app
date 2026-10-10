import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ReviewScope ABSA",
  description: "Analisis sentimen berbasis aspek untuk ulasan aplikasi Shopee",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="id">
      <body>{children}</body>
    </html>
  );
}
