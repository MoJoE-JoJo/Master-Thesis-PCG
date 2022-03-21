package gym;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;
import java.util.List;

public class StepReturnType {
    int[][][] state;
    float reward;
    boolean done;
    HashMap<String,String> info;
    float marioPosition;

    public byte[] getState(){
        // Set up a ByteBuffer called intBuffer
        ByteBuffer intBuffer = ByteBuffer.allocate(4*16*16*1); // 4 bytes in an int
        intBuffer.order(ByteOrder.LITTLE_ENDIAN); // Java's default is big-endian

        // Copy ints from intArray into intBuffer as bytes
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++){
                for (int k = 0; k < 1; k++){
                    intBuffer.putInt(state[i][j][k]);
                }
            }
        }
        // Convert the ByteBuffer to a byte array and return it
        byte[] byteArray = intBuffer.array();
        return byteArray;
    }

    public float getReward(){
        return reward;
    }

    public boolean getDone(){
        return done;
    }

    public HashMap<String,String> getInfo(){
        return info;
    }

    public float getMarioPosition() {
        return marioPosition;
    }
}
