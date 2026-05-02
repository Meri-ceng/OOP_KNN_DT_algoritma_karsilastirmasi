package ml.algorithm;

import ml.model.UserRecord;
import java.util.*;

/**
 * Karar Ağacı (Decision Tree) sınıflandırıcısı.
 * Information Gain (Gini impurity tabanlı) kullanarak özyinelemeli ağaç inşa eder.
 */
public class DecisionTreeClassifier extends BaseAlgorithm {

    // -----------------------------------------------------------------------
    // İç sınıf: Ağaç düğümü
    // -----------------------------------------------------------------------
    private static class Node {
        // Yaprak düğüm alanları
        boolean isLeaf;
        String  leafLabel;

        // İç düğüm alanları
        int    featureIndex;   // Hangi özelliğe göre dallanıyor (0=gender,1=lineNet,2=brand)
        double threshold;      // Eşik değeri
        Node   leftChild;      // feature <= threshold
        Node   rightChild;     // feature >  threshold

        // Yaprak yapıcı
        Node(String label) {
            this.isLeaf    = true;
            this.leafLabel = label;
        }

        // İç düğüm yapıcı
        Node(int featureIndex, double threshold, Node left, Node right) {
            this.isLeaf       = false;
            this.featureIndex = featureIndex;
            this.threshold    = threshold;
            this.leftChild    = left;
            this.rightChild   = right;
        }
    }

    // -----------------------------------------------------------------------
    private int maxDepth;
    private int minSamplesSplit;
    private Node root;

    public DecisionTreeClassifier(int maxDepth, int minSamplesSplit) {
        this.maxDepth        = maxDepth;
        this.minSamplesSplit = minSamplesSplit;
    }

    public int getMaxDepth()         { return maxDepth; }
    public void setMaxDepth(int d)   { this.maxDepth = d; }
    public int getMinSamplesSplit()  { return minSamplesSplit; }
    public void setMinSamplesSplit(int m) { this.minSamplesSplit = m; }

    @Override
    public String getName() {
        return "Decision Tree (maxDepth=" + maxDepth + ")";
    }

    @Override
    public void train(List<UserRecord> trainingData) {
        this.trainingData = new ArrayList<>(trainingData);
        root = buildTree(trainingData, 0);
    }

    @Override
    public String predict(UserRecord user) {
        return traverse(root, user.getFeatureVector());
    }

    // -----------------------------------------------------------------------
    // Özyinelemeli ağaç inşası
    // -----------------------------------------------------------------------
    private Node buildTree(List<UserRecord> data, int depth) {
        // Durma koşulları
        if (data.isEmpty()) return new Node("Unknown");
        if (depth >= maxDepth || data.size() < minSamplesSplit || isPure(data)) {
            return new Node(majorityVote(extractLabels(data)));
        }

        // En iyi bölmeyi bul
        BestSplit best = findBestSplit(data);
        if (best == null) {
            return new Node(majorityVote(extractLabels(data)));
        }

        List<UserRecord> leftData  = new ArrayList<>();
        List<UserRecord> rightData = new ArrayList<>();
        for (UserRecord r : data) {
            if (r.getFeatureVector()[best.featureIndex] <= best.threshold) {
                leftData.add(r);
            } else {
                rightData.add(r);
            }
        }

        // Bölme ilerleme sağlamıyorsa yaprak yap
        if (leftData.isEmpty() || rightData.isEmpty()) {
            return new Node(majorityVote(extractLabels(data)));
        }

        Node left  = buildTree(leftData,  depth + 1);
        Node right = buildTree(rightData, depth + 1);
        return new Node(best.featureIndex, best.threshold, left, right);
    }

    // -----------------------------------------------------------------------
    // En iyi bölmeyi Gini Impurity ile bul
    // -----------------------------------------------------------------------
    private static class BestSplit {
        int    featureIndex;
        double threshold;
        double gain;
        BestSplit(int f, double t, double g) { featureIndex=f; threshold=t; gain=g; }
    }

    private BestSplit findBestSplit(List<UserRecord> data) {
        int numFeatures = data.get(0).getFeatureVector().length;
        double parentGini = giniImpurity(data);
        BestSplit best = null;

        for (int f = 0; f < numFeatures; f++) {
            Set<Double> thresholds = new TreeSet<>();
            for (UserRecord r : data) thresholds.add(r.getFeatureVector()[f]);

            for (double threshold : thresholds) {
                List<UserRecord> left  = new ArrayList<>();
                List<UserRecord> right = new ArrayList<>();
                for (UserRecord r : data) {
                    if (r.getFeatureVector()[f] <= threshold) left.add(r);
                    else right.add(r);
                }
                if (left.isEmpty() || right.isEmpty()) continue;

                double weightedGini = (left.size()  * giniImpurity(left)  +
                                       right.size() * giniImpurity(right)) / data.size();
                double gain = parentGini - weightedGini;

                if (best == null || gain > best.gain) {
                    best = new BestSplit(f, threshold, gain);
                }
            }
        }
        return best;
    }

    private double giniImpurity(List<UserRecord> data) {
        if (data.isEmpty()) return 0.0;
        Map<String, Integer> counts = new HashMap<>();
        for (UserRecord r : data) counts.merge(r.getCategory(), 1, Integer::sum);
        double impurity = 1.0;
        int n = data.size();
        for (int count : counts.values()) {
            double p = (double) count / n;
            impurity -= p * p;
        }
        return impurity;
    }

    private boolean isPure(List<UserRecord> data) {
        String first = data.get(0).getCategory();
        for (UserRecord r : data) if (!r.getCategory().equals(first)) return false;
        return true;
    }

    private List<String> extractLabels(List<UserRecord> data) {
        List<String> labels = new ArrayList<>();
        for (UserRecord r : data) labels.add(r.getCategory());
        return labels;
    }

    private String traverse(Node node, double[] features) {
        if (node.isLeaf) return node.leafLabel;
        if (features[node.featureIndex] <= node.threshold) {
            return traverse(node.leftChild, features);
        } else {
            return traverse(node.rightChild, features);
        }
    }
}
