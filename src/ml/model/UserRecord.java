package ml.model;

/**
 * Veri kümesindeki her satırı temsil eden kapsüllenmiş model sınıfı.
 * Tüm alanlar private, erişim sadece getter metodları ile sağlanır.
 */
public class UserRecord {
    private final String clientCode;
    private final String gender;
    private final double lineNetTotal;
    private final String brand;
    private final String brandCode;
    private final String category;

    // Normalize edilmiş sayısal değerler (PreProcessor tarafından set edilir)
    private double encodedGender;
    private double normalizedLineNetTotal;
    private double encodedBrandCode;

    public UserRecord(String clientCode, String gender, double lineNetTotal,
                      String brand, String brandCode, String category) {
        this.clientCode = clientCode;
        this.gender = gender;
        this.lineNetTotal = lineNetTotal;
        this.brand = brand;
        this.brandCode = brandCode;
        this.category = category;
    }

    // Getters
    public String getClientCode()         { return clientCode; }
    public String getGender()             { return gender; }
    public double getLineNetTotal()       { return lineNetTotal; }
    public String getBrand()              { return brand; }
    public String getBrandCode()          { return brandCode; }
    public String getCategory()           { return category; }
    public double getEncodedGender()      { return encodedGender; }
    public double getNormalizedLineNetTotal() { return normalizedLineNetTotal; }
    public double getEncodedBrandCode()   { return encodedBrandCode; }

    // Setters (sadece PreProcessor kullanır)
    public void setEncodedGender(double v)           { this.encodedGender = v; }
    public void setNormalizedLineNetTotal(double v)  { this.normalizedLineNetTotal = v; }
    public void setEncodedBrandCode(double v)        { this.encodedBrandCode = v; }

    /**
     * Özellik vektörü: [encodedGender, normalizedLineNetTotal, encodedBrandCode]
     */
    public double[] getFeatureVector() {
        return new double[]{encodedGender, normalizedLineNetTotal, encodedBrandCode};
    }

    @Override
    public String toString() {
        return String.format("UserRecord{client=%s, gender=%s, lineNetTotal=%.2f, brand=%s, category=%s}",
                clientCode, gender, lineNetTotal, brand, category);
    }
}
