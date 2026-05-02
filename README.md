# ProLab-II · KNN & Decision Tree — Java Projesi

## Proje Yapısı

```
MLProject/
├── pom.xml                          ← Maven build dosyası
├── gui/
│   └── index.html                   ← Web arayüzü (herhangi bir tarayıcıda açılır)
└── src/ml/
    ├── Main.java                    ← Giriş noktası
    ├── model/
    │   └── UserRecord.java          ← Kapsüllenmiş veri modeli
    ├── algorithm/
    │   ├── IClassifier.java         ← Arayüz (Interface)
    │   ├── BaseAlgorithm.java       ← Soyut temel sınıf (Kalıtım)
    │   ├── KNNClassifier.java       ← K-En Yakın Komşu
    │   └── DecisionTreeClassifier.java ← Karar Ağacı (Gini, Recursive)
    ├── utils/
    │   ├── DataLoader.java          ← xlsx okuyucu (Apache POI — forum onaylı)
    │   └── PreProcessor.java        ← Encoding + Min-Max Normalizasyon
    └── evaluation/
        └── Evaluator.java           ← Doğruluk, hata matrisi, süre raporu
```

## Derleme & Çalıştırma

### Gereksinimler
- Java 17+
- Maven 3.8+

### Adımlar
```bash
# 1. Projeyi derle
cd MLProject
mvn package -q

# 2. Çalıştır (xlsx dosyasını aynı klasöre koyun)
java -jar target/ml-classifier.jar MarketSalesKocaeli.xlsx
```

### Web Arayüzü
```
gui/index.html dosyasını tarayıcıda açın.
xlsx veya csv dosyasını yükleyip "Modeli Çalıştır" butonuna tıklayın.
```

## OOP Mimarisi

| Prensip | Sınıf / Yapı |
|---|---|
| **Encapsulation** | `UserRecord` — tüm alanlar `private`, sadece getter |
| **Interface** | `IClassifier` — `train()`, `predict()`, `getName()` |
| **Inheritance** | `KNNClassifier`, `DecisionTreeClassifier` → `BaseAlgorithm` |
| **Polymorphism** | `List<IClassifier>` ile her iki algoritma tek döngüde çalışır |
| **Abstraction** | `BaseAlgorithm` abstract sınıfı ortak metotları toplar |

## Kütüphane Notu
- ML algoritmaları sıfırdan kodlanmıştır (Weka, Scikit-learn KULLANILMADI).
- Apache POI yalnızca `.xlsx` dosya okuma için kullanılmıştır (forum onaylı).

## Grup
- Üye 1:
- Üye 2:
