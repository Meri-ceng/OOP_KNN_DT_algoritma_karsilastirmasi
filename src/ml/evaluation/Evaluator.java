package ml.evaluation;

import ml.algorithm.BaseAlgorithm;
import ml.algorithm.IClassifier;
import ml.model.UserRecord;

import java.util.*;

/**
 * İki algoritmanın sonuçlarını karşılaştırıp doğruluk ve hız raporu hazırlayan sınıf.
 */
public class Evaluator {

    public static class EvaluationResult {
        public final String  algorithmName;
        public final double  accuracy;          // %
        public final long    trainingTimeMs;
        public final long    predictionTimeMs;
        public final Map<String, Map<String, Integer>> confusionMatrix;
        public final int     testSize;
        public final int     correctCount;

        public EvaluationResult(String name, double accuracy, long trainMs, long predMs,
                                Map<String, Map<String, Integer>> cm, int testSize, int correct) {
            this.algorithmName   = name;
            this.accuracy        = accuracy;
            this.trainingTimeMs  = trainMs;
            this.predictionTimeMs= predMs;
            this.confusionMatrix = cm;
            this.testSize        = testSize;
            this.correctCount    = correct;
        }
    }

    /**
     * Algoritmayı eğitir, test eder ve sonuçları döndürür.
     * @param classifier IClassifier implementasyonu
     * @param trainData  Eğitim verisi
     * @param testData   Test verisi
     */
    public EvaluationResult evaluate(IClassifier classifier,
                                     List<UserRecord> trainData,
                                     List<UserRecord> testData) {
        // --- Eğitim süresi ölçümü ---
        long trainStart = System.currentTimeMillis();
        classifier.train(trainData);
        long trainEnd   = System.currentTimeMillis();

        // --- Tahmin süresi ölçümü ---
        long predStart  = System.currentTimeMillis();
        int correct = 0;
        List<String> predictions = new ArrayList<>();
        for (UserRecord r : testData) {
            String pred = classifier.predict(r);
            predictions.add(pred);
            if (pred.equalsIgnoreCase(r.getCategory())) correct++;
        }
        long predEnd = System.currentTimeMillis();

        double accuracy = (correct * 100.0) / testData.size();

        // Hata matrisi
        Map<String, Map<String, Integer>> cm = buildCM(testData, predictions);

        return new EvaluationResult(
                classifier.getName(),
                accuracy,
                trainEnd - trainStart,
                predEnd  - predStart,
                cm,
                testData.size(),
                correct
        );
    }

    private Map<String, Map<String, Integer>> buildCM(List<UserRecord> testData,
                                                       List<String> predictions) {
        Map<String, Map<String, Integer>> cm = new LinkedHashMap<>();
        for (int i = 0; i < testData.size(); i++) {
            String actual    = testData.get(i).getCategory();
            String predicted = predictions.get(i);
            cm.computeIfAbsent(actual, k -> new LinkedHashMap<>())
              .merge(predicted, 1, Integer::sum);
        }
        return cm;
    }

    /**
     * Konsola okunabilir rapor yazdırır.
     */
    public void printReport(EvaluationResult result) {
        System.out.println("\n========================================");
        System.out.println("  " + result.algorithmName);
        System.out.println("========================================");
        System.out.printf("  Doğruluk       : %.2f%%  (%d / %d)%n",
                result.accuracy, result.correctCount, result.testSize);
        System.out.printf("  Eğitim Süresi  : %d ms%n", result.trainingTimeMs);
        System.out.printf("  Tahmin Süresi  : %d ms%n", result.predictionTimeMs);

        System.out.println("\n  -- Hata Matrisi --");
        // Tüm kategorileri topla
        Set<String> allLabels = new TreeSet<>();
        for (Map.Entry<String, Map<String, Integer>> e : result.confusionMatrix.entrySet()) {
            allLabels.add(e.getKey());
            allLabels.addAll(e.getValue().keySet());
        }
        // Başlık satırı
        System.out.printf("  %-16s", "Gerçek\\Tahmin");
        for (String lbl : allLabels) System.out.printf("%-14s", shorten(lbl));
        System.out.println();
        // Satırlar
        for (String actual : allLabels) {
            System.out.printf("  %-16s", shorten(actual));
            Map<String, Integer> row = result.confusionMatrix.getOrDefault(actual, new HashMap<>());
            for (String pred : allLabels) {
                System.out.printf("%-14d", row.getOrDefault(pred, 0));
            }
            System.out.println();
        }
        System.out.println("========================================\n");
    }

    private String shorten(String s) {
        return s.length() > 12 ? s.substring(0, 12) : s;
    }
}
