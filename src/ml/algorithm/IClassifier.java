package ml.algorithm;

import ml.model.UserRecord;
import java.util.List;

/**
 * Tüm sınıflandırma algoritmalarının uyması gereken standart arayüz.
 * Polymorphism sayesinde KNN ve DecisionTree aynı referans tipiyle kullanılabilir.
 */
public interface IClassifier {
    /**
     * Algoritmayı eğitim verisiyle eğitir / kuralları oluşturur.
     * @param trainingData Eğitim kayıtları
     */
    void train(List<UserRecord> trainingData);

    /**
     * Tek bir kullanıcı için kategori tahmini döndürür.
     * @param user Tahmin yapılacak kayıt
     * @return Tahmin edilen kategori adı
     */
    String predict(UserRecord user);

    /**
     * Algoritmanın adını döndürür (Evaluator raporlamada kullanır).
     */
    String getName();
}
