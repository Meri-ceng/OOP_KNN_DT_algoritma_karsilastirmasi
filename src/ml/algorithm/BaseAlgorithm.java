package ml.algorithm;

import ml.model.UserRecord;
import java.util.*;

/**
 * Her iki algoritmanın ortak metodlarını barındıran soyut temel sınıf.
 * Kalıtım (Inheritance) örneği: KNNClassifier ve DecisionTreeClassifier bu sınıftan türer.
 */
public abstract class BaseAlgorithm implements IClassifier {

    protected List<UserRecord> trainingData;

    /**
     * Test seti üzerinde genel doğruluk oranını hesaplar.
     * @param testData  Test kayıtları
     * @return Doğruluk yüzdesi [0-100]
     */
    public double calculateAccuracy(List<UserRecord> testData) {
        int correct = 0;
        for (UserRecord record : testData) {
            String predicted = predict(record);
            if (predicted.equalsIgnoreCase(record.getCategory())) {
                correct++;
            }
        }
        return (correct * 100.0) / testData.size();
    }

    /**
     * Hata matrisini (Confusion Matrix) hesaplar.
     * @param testData Test kayıtları
     * @return Map< gerçek, Map< tahmin, sayı > >
     */
    public Map<String, Map<String, Integer>> buildConfusionMatrix(List<UserRecord> testData) {
        Map<String, Map<String, Integer>> matrix = new LinkedHashMap<>();
        for (UserRecord record : testData) {
            String actual    = record.getCategory();
            String predicted = predict(record);
            matrix.computeIfAbsent(actual, k -> new LinkedHashMap<>())
                  .merge(predicted, 1, Integer::sum);
        }
        return matrix;
    }

    /**
     * Veri setini eğitim / test olarak rastgele böler.
     * @param data       Tüm kayıtlar
     * @param trainRatio Eğitim oranı (örn. 0.8)
     * @return trainRatio=%80 -> index 0: train, index 1: test
     */
    public static List<List<UserRecord>> splitData(List<UserRecord> data, double trainRatio) {
        List<UserRecord> shuffled = new ArrayList<>(data);
        Collections.shuffle(shuffled, new Random(42)); // Tekrarlanabilirlik için sabit seed
        int splitIdx = (int) (shuffled.size() * trainRatio);
        List<List<UserRecord>> result = new ArrayList<>();
        result.add(new ArrayList<>(shuffled.subList(0, splitIdx)));
        result.add(new ArrayList<>(shuffled.subList(splitIdx, shuffled.size())));
        return result;
    }

    /**
     * Bir listeden en sık geçen elemanı (majority vote) döndürür.
     */
    protected String majorityVote(List<String> labels) {
        Map<String, Integer> counts = new HashMap<>();
        for (String label : labels) counts.merge(label, 1, Integer::sum);
        return counts.entrySet().stream()
                .max(Map.Entry.comparingByValue())
                .map(Map.Entry::getKey)
                .orElse("Unknown");
    }
}
