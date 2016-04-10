/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package hcaim;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.TreeMap;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author guandaline
 */
public abstract class Hierarchy {

    public Instances base;
    public HashMap model;
    public Integer index_attr;
    public Attribute attr;
    public HashMap count_class = null;
    public Integer num_levels;
    public HashMap weights;
    public HashMap intervals;

    public abstract TreeMap discretizeAttr(Double ini, Double fim);

    public void run() {
        this.countNumLevels();
        this.calcWeightLevels();

        this.model = new HashMap<Integer, TreeMap>();

        for (int i = 0; i < this.base.numAttributes(); i++) {
            System.out.println(i);

            int type = this.base.attribute(i).type();

            if (type != Attribute.NUMERIC) {
                continue;
            }

            this.count_class = new HashMap<Integer, HashMap>();
            this.index_attr = i;
            this.attr = this.base.attribute(i);
            this.base.sort(i);

            TreeMap cut_points = this.discretizeAttr(Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

            if (cut_points == null) {
                cut_points = new TreeMap();
            }

            cut_points.put(Double.NEGATIVE_INFINITY, 1);
            cut_points.put(Double.POSITIVE_INFINITY, 1);

            this.model.put(i, cut_points);
            this.count_class = null;
        }
    }

    public String removeRaizNode(String class_name) {
        return class_name.replace("R.", "");
    }

    public String[] splitClass(String class_name) {
        class_name = this.removeRaizNode(class_name);

        return class_name.split("\\.");
    }

    public String getClassAtLevel(String classe, Integer level) {
        String[] parts = this.splitClass(classe);
        String class_at_level = "";

        if (parts.length <= level || level == 0) {
            return this.removeRaizNode(classe);
        }

        for (int i = 0; i < level; i++) {
            class_at_level = class_at_level + parts[i] + ".";
        }

        return class_at_level.substring(0, class_at_level.length() - 1);
    }

    public HashMap calcWeight(Integer h) {
        HashMap weights = new HashMap<Integer, Double>();

        for (int l = 1; l <= h; l++) {
            Double weight = (h.doubleValue() - l + 1) * (2 / (h.doubleValue() * (h.doubleValue() + 1)));
            weights.put(l, weight);
        }

        return weights;
    }

    public Integer getHeightClass(String classe) {
        return classe.split("\\.").length;
    }

    public Integer returnDeeperClassLevel(HashMap classes) {
        Integer height = 0;
        Iterator ite = classes.keySet().iterator();

        while (ite.hasNext()) {
            String class_name = (String) ite.next();
            Integer h = this.getHeightClass(class_name);

            if (h > height) {
                height = h;
            }
        }

        return height;
    }

    public void calcWeightLevels() {
        this.weights = new HashMap<Integer, HashMap>();

        for (int level = 1; level <= this.num_levels; level++) {
            this.weights.put(level, this.calcWeight(level));
        }
    }

    public Double returnWeight(Integer level, Integer depth) {
        HashMap weights_level = (HashMap) this.weights.get(depth);
        return (Double) weights_level.get(level);
    }

    public Integer returnMaxClassOccurrence(HashMap classies) {
        Integer max = Integer.MIN_VALUE;
        Iterator it = classies.keySet().iterator();

        while (it.hasNext()) {
            String key = (String) it.next();
            Integer value = (Integer) classies.get(key);

            if (value > max) {
                max = value;
            }
        }

        return max;
    }

    public Integer returnIntervalNumInstances(HashMap classies) {
        Integer count = 0;
        Iterator it = classies.keySet().iterator();

        while (it.hasNext()) {
            count = count + (Integer) classies.get((String) it.next());
        }

        return count;
    }

    public ArrayList returnNominalValues(Integer attr) {
        TreeMap schema = (TreeMap) this.model.get(attr);
        Iterator it = schema.keySet().iterator();
        Double left = null;
        ArrayList values = new ArrayList();

        while (it.hasNext()) {
            Double right = (Double) it.next();

            if (left != null) {
                values.add("[" + left + ", " + right + "]");
            }

            left = right;
        }

        return values;
    }

    public String returnIntervalValue(Integer index_attr, Double valor) {

        TreeMap schema = (TreeMap) this.model.get(index_attr);
        Iterator it_schema = schema.keySet().iterator();
        Double left = null;

        while (it_schema.hasNext()) {
            Double right = (Double) it_schema.next();

            if (left != null && (left < valor) && (valor <= right)) {
                return "[" + left + ", " + right + "]";
            }

            left = right;
        }

        return null;
    }

    public void countNumLevels() {
        this.num_levels = 0;

        for (int i = 0; i < this.base.numInstances(); i++) {

            Instance ins = this.base.instance(i);
            String class_name = removeRaizNode(this.base.classAttribute().value((int) ins.classValue()));
            String[] parts = class_name.split("\\.");

            if (this.num_levels < parts.length) {
                this.num_levels = parts.length;
            }
        }
    }

    public TreeMap returnPotentialCutPoints(Integer attr, Double begin, Double end) {
        TreeMap points = new TreeMap();
        Instance instances = this.base.instance(0);
        Double previous_class = (Double) instances.classValue();
        Double left = (Double) instances.valueSparse(attr);

        for (int j = 0; j < this.base.numInstances(); j++) {
            instances = this.base.instance(j);
            Double right = (Double) instances.valueSparse(attr);

            if (instances.classValue() != previous_class && left != instances.valueSparse(attr)) {
                previous_class = (Double) instances.classValue();
                Double point = (right + left) / 2;

                if (begin < point && point < end) {
                    points.put(point, 1);
                }
            }

            left = right;
        }

        return points;
    }
    
    private ArrayList returnBaseAttrs(Instances base){
        ArrayList attrs = new ArrayList();
        
         for (int i = 0; i < base.numAttributes(); i++) {
            int type = base.attribute(i).type();

            if (type == Attribute.NUMERIC) {
                String name = base.attribute(i).name();
                ArrayList values = this.returnNominalValues(i);
                attrs.add(new Attribute(name, values, i));
            } else {
                attrs.add(base.attribute(i));
            }
        }
        
        return attrs;
    }

    public Instances createData(Instances base) {
        String relation = base.relationName() + ";Discretizer:Hcaim";
        ArrayList attrs = this.returnBaseAttrs(base);
        Instances data = new Instances(relation, attrs, 0);
        
        for (int i = 0; i < base.numInstances(); i++) {
            Instance ins = base.instance(i);
            double[] values_instance = new double[base.numAttributes()];

            for (int index = 0; index < base.numAttributes(); index++) {
                Double value = ins.value(index);
                int type = base.attribute(index).type();

                if (type == Attribute.NUMERIC) {
                    String interval = this.returnIntervalValue(index, value);
                    Attribute new_attr = (Attribute) attrs.get(index);
                    Integer index_value = new_attr.indexOfValue(interval);
                    value = index_value.doubleValue();
                }

                values_instance[index] = value;
            }

            data.add(i, new DenseInstance(1.0, values_instance));
        }

        return data;
    }

    public HashMap countClassInterval(Double begin, Double end, Integer level) {
        HashMap classies = new HashMap<Integer, HashMap>();

        for (int j = 0; j < this.base.numInstances(); j++) {
            Instance instance = this.base.instance(j);
            Double value = instance.value(this.attr);

            if (begin < value && value <= end) {
                String class_name = this.base.classAttribute().value((int) instance.classValue());
                String class_name_level = this.getClassAtLevel(class_name, level);

                Integer count = classies.containsKey(class_name_level) ? (Integer) classies.get(class_name_level) : 0;
                classies.put(class_name_level, count + 1);
            }
        }

        return classies;
    }
}
