package ml;

import ml.algorithm.*;
import ml.evaluation.Evaluator;
import ml.model.UserRecord;
import ml.utils.DataLoader;
import ml.utils.PreProcessor;

import java.util.List;

/**
 * Ana çalıştırma sınıfı.
 * Veri yükleme → Ön işleme → Eğitim/Test bölme → Model eğitimi → Değerlendirme
 */
public class Main {

    public static void main(String[] args) {
        // Dosya yolu komut satırından veya sabit tanımlı
        String filePath = (args.length > 0) ? args[0] : "MarketSalesKocaeli.xlsx";

        System.out.println("=== ProLab-II: KNN & Decision Tree Karşılaştırması ===");
        System.out.println("Veri dosyası: " + filePath);

        // 1. Veri Yükleme
        DataLoader loader = new DataLoader();
        List<UserRecord> allData;
        try {
            allData = loader.load(filePath);
        } catch (Exception e) {
            System.err.println("HATA: Veri dosyası okunamadı -> " + e.getMessage());
            return;
        }
        System.out.printf("Yüklenen kayıt: %d  |  Atlanan satır: %d%n",
                allData.size(), loader.getSkippedRows());

        // 2. Eğitim / Test Bölme (%80 - %20)
        List<List<UserRecord>> splits = BaseAlgorithm.splitData(allData, 0.80);
        List<UserRecord> trainData = splits.get(0);
        List<UserRecord> testData  = splits.get(1);
        System.out.printf("Eğitim: %d  |  Test: %d%n", trainData.size(), testData.size());

        // 3. Ön İşleme (fit eğitim verisine, transform her ikisine)
        PreProcessor pp = new PreProcessor();
        pp.fit(trainData);
        pp.transform(trainData);
        pp.transform(testData);
        System.out.println("Ön işleme tamamlandı. Marka sayısı: " + pp.getBrandCount());

        // 4. Algoritmalar (Polymorphism: IClassifier listesi)
        List<IClassifier> classifiers = List.of(
                new KNNClassifier(5),
                new DecisionTreeClassifier(10, 5)
        );

        // 5. Değerlendirme
        Evaluator evaluator = new Evaluator();
        for (IClassifier clf : classifiers) {
            Evaluator.EvaluationResult result = evaluator.evaluate(clf, trainData, testData);
            evaluator.printReport(result);
        }
    }
}
