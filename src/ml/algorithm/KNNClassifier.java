package ml.algorithm;

import ml.model.UserRecord;
import java.util.*;

/**
 * K-En Yakın Komşu (K-Nearest Neighbours) sınıflandırıcısı.
 * BaseAlgorithm'dan kalıtım alır, IClassifier arayüzünü implemente eder.
 */
public class KNNClassifier extends BaseAlgorithm {

    private int k;

    public KNNClassifier(int k) {
        this.k = k;
    }

    public int getK() { return k; }
    public void setK(int k) { this.k = k; }

    @Override
    public String getName() {
        return "KNN (k=" + k + ")";
    }

    /**
     * Eğitim verilerini saklar (KNN lazy learner'dır, train aşamasında işlem yoktur).
     */
    @Override
    public void train(List<UserRecord> trainingData) {
        this.trainingData = new ArrayList<>(trainingData);
    }

    /**
     * Yeni kayıt için tüm eğitim kayıtlarına Öklid mesafesi hesaplar,
     * en yakın K komşunun kategorisine göre majority vote yapar.
     */
    @Override
    public String predict(UserRecord user) {
        double[] queryVec = user.getFeatureVector();

        // PriorityQueue: mesafeye göre büyükten küçüğe (max-heap), en fazla K eleman tutar
        PriorityQueue<double[]> maxHeap = new PriorityQueue<>(
                k, (a, b) -> Double.compare(b[0], a[0])
        );

        for (int i = 0; i < trainingData.size(); i++) {
            UserRecord record = trainingData.get(i);
            double dist = euclideanDistance(queryVec, record.getFeatureVector());
            if (maxHeap.size() < k) {
                maxHeap.offer(new double[]{dist, i});
            } else if (dist < maxHeap.peek()[0]) {
                maxHeap.poll();
                maxHeap.offer(new double[]{dist, i});
            }
        }

        List<String> neighborLabels = new ArrayList<>();
        for (double[] entry : maxHeap) {
            neighborLabels.add(trainingData.get((int) entry[1]).getCategory());
        }
        return majorityVote(neighborLabels);
    }

    /**
     * İki özellik vektörü arasındaki Öklid mesafesini hesaplar.
     * distance = sqrt( sum( (a_i - b_i)^2 ) )
     */
    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }
}
