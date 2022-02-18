package gym;

import engine.core.*;
import engine.helper.Assets;
import engine.helper.GameStatus;
import engine.helper.MarioActions;
import py4j.GatewayServer;

import javax.swing.*;
import java.awt.*;
import java.awt.image.VolatileImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;

public class MarioGymManager {
    static ArrayList<MarioGym> gyms = new ArrayList<>();


    public static void main(String[] args) {
        MarioGymManager gymManager = new MarioGymManager();
        // app is now the gateway.entry_point
        GatewayServer server = new GatewayServer(gymManager);
        server.start();
        System.out.println("Gateway Started");
    }

    public static void initGym(String paramLevel, String imageDirectory, int timer, int paramMarioState, boolean visual){
        MarioGym newGym = new MarioGym();
        newGym.init(paramLevel, imageDirectory, timer, paramMarioState, visual);
        gyms.add(newGym);
    }

    public static StepReturnType step(int gymID, boolean left, boolean right, boolean down, boolean speed, boolean jump){
        return gyms.get(gymID).step(left, right, down, speed, jump);
    }

    public static StepReturnType reset(int gymID, boolean visual){
        return gyms.get(gymID).reset(visual);
    }

    public static void render(int gymID){
        gyms.get(gymID).render();
    }

    public static void setLevel(int gymID, String levelParam){
        gyms.get(gymID).setLevel(levelParam);
    }
/*
    public static void playGame(int gymID, String level, int time, int marioState, boolean visuals){
        gyms.get(gymID).playGame(level, time, marioState, visuals);
    }

    public static void agentInput(boolean left, boolean right, boolean down, boolean speed, boolean jump){
        boolean[] actions = new boolean[]{left, right, down, speed, jump};
        agent.setActions(actions);
    }

    private static String getLevel(String filepath) {
        String content = "";
        try {
            content = new String(Files.readAllBytes(Paths.get(filepath)));
        } catch (IOException e) {
        }
        return content;
    }

    private static void printResults(MarioResult result) {
        System.out.println("****************************************************************");
        System.out.println("Game Status: " + result.getGameStatus().toString() +
                " Percentage Completion: " + result.getCompletionPercentage());
        System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins() +
                " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
        System.out.println("Mario State: " + result.getMarioMode() +
                " (Mushrooms: " + result.getNumCollectedMushrooms() + " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
        System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp() +
                " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() +
                " Falls: " + result.getKillsByFall() + ")");
        System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps() +
                " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
        System.out.println("****************************************************************");
    }
 */
}
