package gym;

import java.util.List;

public class StepReturnType {
    int[][] state;
    int reward;
    boolean done;
    List<String> info;

    public int[][] getState(){
        return state;
    }

    public int getReward(){
        return reward;
    }

    public boolean getDone(){
        return done;
    }

    public List<String> getInfo(){
        return info;
    }
}
