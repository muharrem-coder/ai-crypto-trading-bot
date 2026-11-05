# quant-core kullanım kılavuzu

Bu kılavuz sistemi kendi bilgisayarında çalıştırmak, terminalden izlemek ve sırasıyla PAPER, TESTNET ve LIVE modlarına geçmek için yazıldı. Komutlar Windows PowerShell içindir. Hepsini proje klasöründe (`quant-core`) çalıştır.

## Modlar

| Mod | Fiyat ve haber | Emirler | Para |
|---|---|---|---|
| `PAPER` | Binance'in gerçek canlı verisi | Bilgisayarındaki sanal borsa (komisyon ve kayma dahil) | Yok, sanal 10.000 USDT |
| `TESTNET` | Borsanın test ağı | Test ağına gerçek CCXT emirleri | Test ağının oyun parası |
| `LIVE` | Borsanın canlı verisi | Gerçek borsaya gerçek emirler | Gerçek para |

Sıra hep aynıdır: önce günlerce PAPER, sonra TESTNET, en son LIVE.

## 1. Bugün PAPER ile başlat (Docker ile, önerilen yol)

Gerekenler: Docker Desktop kurulu ve açık olmalı.

1. Ayar dosyasını oluştur:

   ```powershell
   copy .env.example .env
   notepad .env
   ```

2. `.env` içinde şunları değiştir:
   - `POSTGRES_PASSWORD`: kendine bir şifre koy. Harf ve rakam kullan, özel karakter koyma.
   - `ANTHROPIC_API_KEY`: Claude API anahtarın. Boş bırakırsan sistem çalışır ama her karar HOLD olur, yani hiç işlem açılmaz.
   - `CLAUDE_MODEL`: kullanmak istediğin Claude modelinin kimliği. Preflight aracı bu kimliğin geçerli olup olmadığını kontrol eder.
   - `API_CONTROL_TOKEN`: terminaldeki MARKET CLOSE ve RESUME butonları bunu ister. Üretmek için:

     ```powershell
     docker run --rm python:3.12-slim python -c "import secrets;print(secrets.token_urlsafe(32))"
     ```

   `TRADING_MODE=PAPER` olarak kalsın. Borsa API anahtarı PAPER'da gerekmez.

3. Sistemi başlat:

   ```powershell
   docker compose up -d --build
   docker compose ps
   ```

   Üç servis de (`timescaledb`, `redis`, `quant-core`) bir dakika içinde `healthy` olmalı. Veritabanı tabloları ilk açılışta uygulama tarafından otomatik kurulur.

4. Uçuş öncesi kontrolü çalıştır:

   ```powershell
   docker compose exec quant-core python -m app.cli.preflight
   ```

   Sonda `HAZIR / READY for PAPER` yazmalı. `[FAIL]` olan satır varsa önce onu düzelt.

## 1b. Docker olmadan çalıştırmak

Docker kullanmak istemezsen Python 3.11 veya üstü yeter. Veritabanı olarak tek dosyalık SQLite kullanılır.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e .
copy .env.example .env
```

`.env` içinde `DATABASE_URL` satırını şöyle değiştir:

```
DATABASE_URL=sqlite+aiosqlite:///./quant.db
```

Sonra:

```powershell
python -m app.cli.preflight
python -m app.api.server
```

SQLite ile TimescaleDB ve Redis kontrolleri uyarı (`[WARN]`) verir. PAPER için sorun değildir. LIVE için Docker ile çalıştır.

## 2. Terminali izle

Tarayıcıda `http://127.0.0.1:8000` adresini aç.

- Sol üstteki rozet sistemin durumunu gösterir. `AUTONOMOUS 24/7` (yeşil) her şey yolunda demektir. `DATA PAUSED` (sarı) emir defteri verisi 2 saniyeden eski olduğu için sinyal üretilmediği anlamına gelir. `HALTED` (kırmızı) kill-switch devrede demektir.
- `API LATENCY` tarayıcı ile sunucu arasındaki gecikmedir.
- Üstteki KPI şeridi bakiyeyi, açık kâr/zararı, son 24 saatin net kârını, ödenen komisyonu, kazanma oranını ve günlük zarar limitine ne kadar yaklaşıldığını gösterir.
- Grafik aktif sembolün mumlarını ve açılan işlemlerin işaretlerini gösterir.
- Sağ üstteki "Claude Decision & Reasoning Log" her kararı, teknik özeti ve Türkçe gerekçeyi gösterir. Kart önce kararla gelir, emir sonucu belli olunca güncellenir (`REDUCED`, `OPENED_PROTECTED` gibi).
- Sağ alt köşede tekilleştirilmiş haberler ve duygu skorları var.
- Alttaki tabloda açık pozisyonlar, borsadaki hard stop ve take-profit seviyeleri, tahmini komisyon ve net PnL yer alır. MARKET CLOSE butonu pozisyonu piyasa emriyle kapatır ve SL/TP emirlerini iptal eder. İlk basışta senden `API_CONTROL_TOKEN` ister; token sadece o sekme açıkken hafızada tutulur.

Bilmen gerekenler:

- Claude her mumda çağrılmaz. Sinyal ancak strateji uyumu ve komisyon kontrolü geçerse gider, saatte en fazla 12 kez. Sakin piyasada saatlerce karar gelmemesi normaldir.
- PAPER'daki sanal hesap bellekte tutulur. Konteyneri yeniden başlatırsan sanal bakiye ve pozisyonlar sıfırlanır, açık kalan işlem kayıtları veritabanında `MANUAL` olarak kapatılır.
- Loglar: `docker compose logs -f quant-core`
- Sağlık durumu: `http://127.0.0.1:8000/api/v1/health`

## 3. TESTNET'e geç

1. Test ağı hesabı aç ve API anahtarı oluştur:
   - Binance vadeli işlem test ağı: `https://testnet.binancefuture.com`
   - Bybit test ağı: `https://testnet.bybit.com`
2. `.env` içinde:

   ```
   TRADING_MODE=TESTNET
   EXCHANGE_ID=binance
   EXCHANGE_API_KEY=<test ağı anahtarı>
   EXCHANGE_API_SECRET=<test ağı gizli anahtarı>
   EXCHANGE_TESTNET=false
   ```

   Bybit kullanıyorsan `EXCHANGE_ID=bybit` yaz. `EXCHANGE_TESTNET` satırına dokunma; TESTNET modunda test ağı kendiliğinden açılır.
3. Kontrol et ve yeniden başlat:

   ```powershell
   docker compose exec quant-core python -m app.cli.preflight --mode TESTNET
   docker compose up -d
   ```

   Ayar değiştiğinde `docker compose up -d` konteyneri yeni `.env` ile yeniden kurar.
4. Test ağında en az birkaç gün izle. Bakman gerekenler: her açılan pozisyonda borsada stop emri var mı, break-even kilidi çalışıyor mu, yeniden başlatmadan sonra pozisyonlar korunuyor mu.

## 4. LIVE (gerçek para) geçişi

Başlamadan önce:

- PAPER ve TESTNET sonuçlarını terminalden ve işlem geçmişinden incele. Komisyon sonrası net sonuç eksi ise LIVE'a geçme.
- Gerçek hesapta küçük bir bakiye ile başla. Kaybetmeyi göze alamayacağın parayı koyma.
- Gerçek borsa API anahtarında "Enable Withdrawals" (para çekme) izni kapalı olsun. Vadeli işlem izni açık olsun. Anahtara IP kısıtlaması ekle.

`.env` değişiklikleri:

```
TRADING_MODE=LIVE
LIVE_TRADING_CONFIRMATION=I_UNDERSTAND_REAL_MONEY_IS_AT_RISK
EXCHANGE_API_KEY=<gerçek anahtar>
EXCHANGE_API_SECRET=<gerçek gizli anahtar>
EXCHANGE_TESTNET=false
API_CONTROL_TOKEN=<en az 32 karakter>
```

Risk limitlerini gevşetme. Varsayılanlar: en fazla 5x kaldıraç, günlük en fazla %3 zarar, işlem başına bakiyenin %0,5'i risk.

Önce kontrol:

```powershell
docker compose exec quant-core python -m app.cli.preflight --mode LIVE --claude-roundtrip
```

`HAZIR / READY for LIVE` yazmadan başlatma. LIVE'da bazı uyarılar hataya dönüşür (IP kısıtlaması yok, saat kayması, Claude gecikmesi gibi). `--claude-roundtrip` birkaç token harcayarak gerçek bir karar çağrısı dener.

Sonra başlat:

```powershell
docker compose up -d
docker compose logs -f quant-core
```

LIVE açılışında sistem kendi kendine şunları yapar. Biri tutmazsa başlamaz:

- Ayar kilitlerini tekrar kontrol eder (onay cümlesi, anahtarlar, token).
- Bakiyenin `RECON_LIVE_MIN_BALANCE` değerinin üstünde olduğunu doğrular.
- Her sembol için izole marjin ve kaldıraç ayarını yapar. Borsada çapraz marjinli ya da 5x üstü bir pozisyon varsa durur.
- Borsadaki mevcut pozisyonları tarar. Stopu olmayana stop koyar, güvenlik kuralına uymayanı kapatır. Bu mutabakat çalışırken de her 60 saniyede bir tekrarlanır.

## 5. Acil durum

Her şeyi durdurmak için:

1. Terminaldeki kırmızı EMERGENCY KILL-SWITCH butonuna bas. Token istemez. Bütün emirleri iptal eder, pozisyonları piyasa emriyle kapatır ve yeni işlem açılmasını engeller. Durum veritabanına yazılır, yeniden başlatmada da kapalı kalır.
2. Terminal açılmıyorsa aynı işi komutla yap:

   ```powershell
   curl.exe -X POST http://127.0.0.1:8000/api/v1/kill-switch
   ```

3. Uygulamanın kendisine ulaşamıyorsan borsanın kendi sitesinden ya da uygulamasından pozisyonları elle kapat, sonra `docker compose stop quant-core` ile botu durdur.

Bot durursa ya da internet giderse açık pozisyonlar korumasız kalmaz. Stop-loss emirleri borsada bekler ve botun çalışmasına bağlı değildir. Bot geri geldiğinde mutabakat servisi pozisyonları tekrar sahiplenir. İz süren stop o andan itibaren yeniden çalışır.

Kill-switch'i tekrar açmak için terminalde RESUME butonuna bas. Adını ve `RESET` kelimesini ister. Sonra token sorar. Önce neden durduğunu loglardan ve terminaldeki uyarı kartlarından anla:

```powershell
docker compose logs --since 1h quant-core
```

## 6. Bakım

- Durdur: `docker compose stop`. Veriler `timescaledb` biriminde kalır.
- Güncelleme sonrası yeniden kur: `docker compose up -d --build`
- Veritabanı yedeği:

  ```powershell
  docker compose exec -T timescaledb pg_dump -U quant -f /tmp/yedek.sql quant
  docker compose cp timescaledb:/tmp/yedek.sql .\yedek.sql
  ```

- Tamamen silmek (bütün geçmiş dahil): `docker compose down -v`. Geri alınamaz.

## Bilinen sınırlar

- Redis servisi kuruluyor ve preflight onu kontrol ediyor, ama bu sürümde kod Redis'i henüz kullanmıyor. Olaylar uygulamanın içinde dağıtılıyor.
- Sistem Binance ve Bybit için yazıldı. Borsa izni kontrolü (para çekme kapalı mı) sadece bu ikisinde otomatik. Başka borsada elle kontrol etmen gerekir.
- Testlerin hepsi sahte borsa, sahte Claude cevapları ve Docker içinde gerçek TimescaleDB ile yapıldı. Test ağına gerçek emir gönderimi senin bilgisayarında, TESTNET adımında ilk kez denenecek.
- Fiyat stopun içinden boşlukla geçerse stop daha kötü bir fiyattan dolar. Hiçbir stop emri bunu engelleyemez.
